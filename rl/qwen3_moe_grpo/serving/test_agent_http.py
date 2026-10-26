"""Real loopback Agent HTTP wiring, scripted model responses (no GPU)."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agent_server import handler_for
from test_candidate_demo import Bundle, response


class AgentHTTPTests(unittest.TestCase):
    def test_auth_sessions_and_server_owned_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = []
            def send(body):
                seen.append(body)
                return response(text='李老师。')
            key = 'test-secret-for-local-scripted-backend'
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(Bundle(), send, key, tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            def call(path, body):
                with urlopen(Request(base + path, headers={'Authorization':'Bearer ' + key},
                    data=json.dumps(body).encode()), timeout=5) as result:
                    return json.load(result)
            try:
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + '/health', timeout=5)
                self.assertEqual(error.exception.code, 401)
                created = call('/sessions', dict(model='qwen3-grpo'))
                path = '/sessions/' + created['session_id'] + '/messages'
                self.assertTrue(call(path, dict(text='怎么称呼？'))['completed'])
                self.assertTrue(call(path, dict(text='你是谁？'))['completed'])
                self.assertEqual(sum(m['role']=='user' for m in seen[-1]['messages']), 2)
                with self.assertRaises(HTTPError) as error:
                    call(path, dict(text='拉群', consent=True))
                self.assertEqual(error.exception.code, 400)
                logs = list(Path(tmp).glob('*.jsonl'))
                self.assertEqual(len(logs), 1)
                self.assertEqual(len(logs[0].read_text().splitlines()), 2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
