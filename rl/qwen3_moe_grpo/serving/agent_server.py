"""Stateful research Agent API around Qwen, business prompts and in-memory tools.

No OS/shell tool, network business tool or real message delivery is registered.
The API is private loopback-only; a fresh random key protects all endpoints.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading

from candidate_demo import run_case
from candidate_mock import MockBusiness
from candidate_prompts import PromptBundle, INDUSTRIES
from protocol import MODELS
from recruiting_demo import chat


class CandidateAgent:
    def __init__(self, bundle, model, send, industry='homecare'):
        if model not in MODELS or industry not in INDUSTRIES:
            raise ValueError('Invalid model or industry')
        self.bundle, self.model, self.send = bundle, model, send
        self.business = MockBusiness(bundle, dict(industry=industry,
            name='李老师' if industry == 'homecare' else '进厂顾问'))
        self.messages, self.turns, self.broken = [], 0, False

    def turn(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError('Message must contain 1..2000 characters')
        if self.broken or self.turns >= 12:
            raise ValueError('Start a new session; this session ended or reached its turn limit')
        self._consent(text)
        result = run_case(dict(id=f'turn-{self.turns+1}', prompt=text), self.model, self.send,
                          self.bundle, business=self.business, messages=self.messages)
        self.turns += 1
        if result['completed']:
            # Only mark an introduction when a successful current-job receipt backs
            # recognizable job facts in the final reply. This is a mock heuristic.
            job = self.business.job(self.business.active)
            answer = result['answer']
            receipt = any(e['kind'] == 'tool' and e['result'].get('status') == 'SUCCESS'
                          and e['result'].get('activeJob', {}).get('jobId') == self.business.active
                          for e in result['trace'])
            if job and receipt and (job['jobName'] in answer or
                    job['category'] in answer and job['district'] in answer):
                self.business.exposed = self.business.active
        else:
            # A failed generation may leave an unfinished tool protocol. Keep its
            # trace for diagnosis, but never replay a partially committed turn.
            self.broken = True
        result['final_context'] = self.business.context()
        return result

    def _consent(self, text):
        # Conservative fixture-only interpretation, not a production NLU gate.
        self.business.consent_job = ''
        if not self.business.active or self.business.exposed != self.business.active:
            return
        if re.search(r'不|别|暂|换|但是|不过|仅|只|停止|取消|不要', text):
            return
        explicit = re.fullmatch(r'[，,。！!\s]*(?:可以[，,\s]*)?(?:请)?(?:帮我|给我)(?:拉群|进群|对接老师)[，,。！!\s]*', text)
        prior = next((m.get('content', '') for m in reversed(self.messages)
                      if m['role'] == 'assistant' and not m.get('tool_calls')), '')
        weak = text.strip('，,。！! \n') in ('可以', '好', '好的', '行', '同意')
        sole_question = bool(re.fullmatch(r'(?:需要|要|可以)?(?:帮您|帮你|为您)?(?:拉群|进群)(?:对接)?(?:吗|可以吗)[？?]', prior.strip()))
        if explicit or weak and sole_question:
            self.business.consent_job = self.business.active


def handler_for(bundle, send, key, trace_dir):
    sessions, gate = {}, threading.Lock()
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)
        def log_message(self, *_):
            pass
        def reply(self, code, data):
            payload = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def authorized(self):
            return secrets.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + key)
        def do_GET(self):
            if not self.authorized():
                return self.reply(401, dict(error='Unauthorized'))
            if self.path == '/health':
                return self.reply(200, dict(agent_ready=True, backend_readiness='check model service separately',
                    prompt_sha256=bundle.sha256, tools_backend='synthetic-in-memory', sessions=len(sessions)))
            return self.reply(404, dict(error='Not found'))
        def do_POST(self):
            if not self.authorized():
                return self.reply(401, dict(error='Unauthorized'))
            if not gate.acquire(blocking=False):
                return self.reply(429, dict(error='Agent busy'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Expected object')
                if self.path == '/sessions':
                    if set(body) - {'model', 'industry'} or len(sessions) >= 32:
                        raise ValueError('Invalid fields or session capacity reached')
                    agent = CandidateAgent(bundle, body.get('model', 'qwen3-grpo'), send, body.get('industry', 'homecare'))
                    ident = secrets.token_hex(16)
                    sessions[ident] = agent
                    return self.reply(201, dict(session_id=ident, model=agent.model, simulated=True))
                match = re.fullmatch(r'/sessions/([0-9a-f]{32})/messages', self.path)
                if not match or match[1] not in sessions:
                    return self.reply(404, dict(error='Session not found'))
                if set(body) != {'text'}:
                    raise ValueError('Only text is accepted; context and consent are application-owned')
                ident, agent = match[1], sessions[match[1]]
                result = agent.turn(body['text'])
                path = trace_dir / (ident + '.jsonl')
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(descriptor, 'a') as output:
                    output.write(json.dumps(result, ensure_ascii=False) + '\n')
                return self.reply(200 if result['completed'] else 422, dict(
                    session_id=ident, completed=result['completed'], answer=result.get('answer'),
                    error=result.get('error'), seconds=result['seconds'], tool_calls=result['tool_calls'],
                    simulated=True, review_required=True))
            except (ValueError, TypeError, KeyError) as error:
                return self.reply(400, dict(error=str(error)))
            finally:
                gate.release()
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-dir', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--trace-dir', type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get('QWEN3_API_KEY', '')
    if len(key) < 24:
        parser.error('Set QWEN3_API_KEY')
    bundle = PromptBundle(args.prompt_dir)
    handler = handler_for(bundle, lambda body: chat(args.base_url, key, body), key, args.trace_dir)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    print(json.dumps(dict(event='agent_ready', port=args.port, prompt_sha256=bundle.sha256)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
