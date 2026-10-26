"""CPU tests of protocol, executor boundaries and HTTP wiring, not GPU quality."""
import copy
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from protocol import parse_output, prepare_request, service_seconds
from recruiting_demo import TOOLS, execute_tool, run_case, CASES, check_case
from serve import handler_for


def response(text):
    message, reason = parse_output(text, TOOLS)
    return dict(choices=[dict(message=message, finish_reason=reason)], usage={}, metrics={})


class ProtocolTests(unittest.TestCase):
    def test_budget_counts_boot_time_and_hard_stop(self):
        self.assertEqual(service_seconds(10, 5.59, 1000, 6400, 1300), 3300)
        self.assertEqual(service_seconds(1, 6, 1000, 6400, 1000), 420)
        for values in [(10, 5.59, 1000, 6401, 1300), (0, 6, 1000, 6400, 1000),
                       (1, 6, 1000, 6400, 1300), (10, float('nan'), 1000, 6400, 1000)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                service_seconds(*values)

    def test_multiple_calls_and_roundtrip(self):
        text = ('<tool_call>{"name":"search_jobs","arguments":{"query":"Java","city":"北京"}}</tool_call>'
                '<tool_call>{"name":"get_recruiting_progress","arguments":{"job_id":"J101"}}</tool_call>')
        message, reason = parse_output(text, TOOLS)
        self.assertEqual(reason, 'tool_calls')
        self.assertEqual(len(message['tool_calls']), 2)
        original = copy.deepcopy(message)
        request = dict(model='qwen3-grpo', tools=TOOLS, messages=[dict(role='user', content='test'), message] +
                       [dict(role='tool', tool_call_id=c['id'], content='{}') for c in message['tool_calls']])
        prepared = prepare_request(request)
        self.assertIsInstance(prepared['messages'][1]['tool_calls'][0]['function']['arguments'], dict)
        self.assertEqual(message, original)

    def test_no_execution_of_reasoning(self):
        message, reason = parse_output('<think><tool_call>{bad}</tool_call></think>你好', TOOLS)
        self.assertEqual(message['content'], '你好')
        self.assertEqual(reason, 'stop')

    def test_malformed_unknown_truncated_rejected(self):
        samples = ['<tool_call>{bad}</tool_call>', '<tool_call>{"name":"search_jobs"}',
                   '<tool_call>{"name":"delete_job","arguments":{}}</tool_call>',
                   '<tool_call>{"name":"search_jobs","arguments":[]}</tool_call>', '<think>unfinished']
        for sample in samples:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                parse_output(sample, TOOLS)
        with self.assertRaises(ValueError):
            parse_output('<tool_call>{"name":"search_jobs","arguments":{}}</tool_call>', TOOLS, True)

    def test_request_limits_and_unknown_fields(self):
        body = dict(model='qwen3-base', messages=[dict(role='user', content='你好')])
        for extra in [dict(stream=True), dict(max_tokens=2048), dict(temperature=float('nan')),
                      dict(model='other'), dict(top_p=0), dict(seed=123), dict(tools='bad')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                prepare_request(dict(body, **extra))

    def test_unmatched_result_rejected(self):
        with self.assertRaises(ValueError):
            prepare_request(dict(model='qwen3-base', messages=[dict(role='tool', tool_call_id='invented', content='{}')]))

    def test_none_choice_never_allows_tool_call(self):
        request = prepare_request(dict(model='qwen3-base', tools=TOOLS, tool_choice='none',
                                       messages=[dict(role='user', content='hello')]))
        with self.assertRaises(ValueError):
            parse_output('<tool_call>{"name":"search_jobs","arguments":{}}</tool_call>', request['tools'])


class BusinessTests(unittest.TestCase):
    def test_fixture_query_and_matching(self):
        jobs = execute_tool('search_jobs', dict(query='Java', city='北京'))['jobs']
        self.assertEqual([j['job_id'] for j in jobs], ['J101'])
        matches = execute_tool('match_candidates', dict(job_id='J101', top_k=2))['candidates']
        self.assertEqual([m['candidate_id'] for m in matches], ['C101', 'C102'])

    def test_scope_and_parameters_are_enforced(self):
        self.assertEqual(execute_tool('get_recruiting_progress', dict(job_id='J900'))['error'], 'FORBIDDEN_OR_NOT_FOUND')
        self.assertEqual(execute_tool('search_jobs', dict(query='Java', city='深圳'))['jobs'], [])
        self.assertEqual(execute_tool('match_candidates', dict(job_id='J101', top_k=True))['error'], 'INVALID_ARGUMENTS')
        self.assertEqual(execute_tool('get_recruiting_progress', dict(job_id='J101', tenant_id='other'))['error'], 'INVALID_ARGUMENTS')
        self.assertEqual(execute_tool('send_invitation', {})['error'], 'UNKNOWN_TOOL')

    def test_tool_result_returns_to_model(self):
        requests = []
        def scripted(body):
            requests.append(copy.deepcopy(body))
            if len(requests) == 1:
                return response('<tool_call>{"name":"get_recruiting_progress","arguments":{"job_id":"J101"}}</tool_call>')
            result = json.loads(body['messages'][-1]['content'])
            self.assertEqual(result['interviewing'], 3)
            return response('J101：12名进入漏斗，8名已联系，3名面试中，1名已录用。')
        result = run_case(CASES[2]['prompt'], 'qwen3-grpo', scripted)
        self.assertTrue(result['completed'])
        checks = check_case(CASES[2], result)
        self.assertTrue(checks['exact_tool_arguments_pass'])
        self.assertTrue(checks['needs_human_review'])

    def test_loop_is_bounded(self):
        call = '<tool_call>{"name":"get_recruiting_progress","arguments":{"job_id":"J101"}}</tool_call>'
        result = run_case('test', 'qwen3-base', lambda _: response(call), max_rounds=2)
        self.assertFalse(result['completed'])
        self.assertEqual(result['tool_calls'], 2)
        self.assertIn('budget exhausted', result['error'])


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class FakeBackend:
            info = {'backend': 'scripted-test-double'}
            def generate(self, request):
                result = response('mock answer')
                result['model'] = request['model']
                return result
        cls.key = 'local-test-secret-not-for-deployment'
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(FakeBackend(), cls.key))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_http_models_and_inference(self):
        headers = {'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'}
        with urlopen(Request(self.url + '/v1/models', headers=headers)) as response_:
            self.assertEqual(len(json.load(response_)['data']), 2)
        for model in ['qwen3-base', 'qwen3-grpo']:
            body = dict(model=model, messages=[dict(role='user', content='test')])
            with urlopen(Request(self.url + '/v1/chat/completions', headers=headers,
                                 data=json.dumps(body).encode())) as response_:
                self.assertEqual(json.load(response_)['model'], model)

    def test_http_requires_auth_and_rejects_stream(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + '/health')
        self.assertEqual(error.exception.code, 401)
        body = dict(model='qwen3-base', messages=[dict(role='user', content='x')], stream=True)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                            headers={'Authorization': 'Bearer '+self.key}))
        self.assertEqual(error.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
