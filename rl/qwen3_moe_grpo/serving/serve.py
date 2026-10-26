"""Single-GPU, loopback-only research server; never executes business tools."""
import argparse
from contextlib import nullcontext
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
import uuid

from protocol import MODELS, parse_output, prepare_request


class Backend:
    def __init__(self, model_path, source_run, max_context):
        os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')
        os.environ.setdefault('UNSLOTH_MOE_BACKEND', 'unsloth_triton')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        from unsloth import FastLanguageModel
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from round2 import restore_adapter
        source_run, model_path = Path(source_run), Path(model_path)
        config = json.loads((source_run / 'config.json').read_text())
        summary = json.loads((source_run / 'summary.json').read_text())
        if not summary.get('completed') or summary.get('optimizer_steps') != 512:
            raise RuntimeError('Expected the completed second-round adapter')
        if config['model_id'] != 'Qwen/Qwen3-30B-A3B' or config['precision'] != 'bf16':
            raise RuntimeError('Unexpected base model or precision')
        if not (model_path / 'config.json').is_file():
            raise RuntimeError('Local base model is missing; this service does not download weights')
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).total_memory / 2**30 < 78:
            raise RuntimeError('Expected one 80GB-class GPU')
        torch.set_num_threads(8)
        self.torch, self.max_context = torch, max_context
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path), max_seq_length=max_context, dtype=torch.bfloat16,
            load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
            fast_inference=False, device_map={'': 0}, trust_remote_code=False)
        if getattr(self.model, 'is_loaded_in_4bit', False) or getattr(self.model, 'is_loaded_in_8bit', False):
            raise RuntimeError('Unexpected quantization')
        self.model = FastLanguageModel.get_peft_model(
            self.model, r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=config['target_modules'],
            bias='none', use_gradient_checkpointing=True, random_state=config['seed'])
        restore_adapter(self.model, source_run / 'adapter')
        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.requires_grad_(False)
        self.info = dict(source_run=str(source_run), model_path=str(model_path), precision='bf16',
                         max_context=max_context, enable_thinking=False, concurrency=1,
                         config_sha256=hashlib.sha256((source_run / 'config.json').read_bytes()).hexdigest(),
                         adapter_tensor_verification=True, optimizer_updates=0)

    def generate(self, request):
        torch = self.torch
        started = time.perf_counter()
        options = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if request['tools']:
            options['tools'] = request['tools']
        prompt = self.tokenizer.apply_chat_template(request['messages'], **options)
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
        n_prompt = inputs.input_ids.shape[1]
        if n_prompt + request['max_tokens'] > self.max_context:
            raise ValueError('Context limit exceeded; shorten history/tool results or max_tokens')

        class FirstToken:
            def __init__(self):
                self.prompt_seen, self.first_at = False, None
            def put(self, value):
                if not self.prompt_seen:
                    self.prompt_seen = True
                elif self.first_at is None:
                    self.first_at = time.perf_counter()
            def end(self):
                pass

        clock = FirstToken()
        sample = request['temperature'] > 0
        sampling = dict(temperature=request['temperature'], top_p=request['top_p'], top_k=20) if sample else {}
        context = self.model.disable_adapter() if request['model'] == 'qwen3-base' else nullcontext()
        torch.cuda.reset_peak_memory_stats()
        with context, torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=request['max_tokens'],
                do_sample=sample, use_cache=True, streamer=clock, **sampling,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        ids = output[0, n_prompt:].tolist()
        truncated = self.tokenizer.eos_token_id not in ids
        # Keep tool tags even if marked as special tokens; strip only EOS/padding.
        text = self.tokenizer.decode([t for t in ids if t not in
            (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)], skip_special_tokens=False)
        elapsed = time.perf_counter() - started
        message, reason = parse_output(text, request['tools'], truncated)
        return dict(id='chatcmpl-' + uuid.uuid4().hex, object='chat.completion', created=int(time.time()),
            model=request['model'], choices=[dict(index=0, message=message, finish_reason=reason)],
            usage=dict(prompt_tokens=n_prompt, completion_tokens=len(ids), total_tokens=n_prompt+len(ids)),
            metrics=dict(server_seconds=elapsed, server_ttft_seconds=(clock.first_at-started if clock.first_at else None),
                         output_tokens_per_second=len(ids)/elapsed,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


def handler_for(backend, api_key):
    gate = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)
        def log_message(self, *_):
            pass  # Do not put business messages or auth headers in HTTP access logs.
        def reply(self, code, data):
            payload = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def authorized(self):
            actual = self.headers.get('Authorization', '')
            return secrets.compare_digest(actual, 'Bearer ' + api_key)
        def do_GET(self):
            if not self.authorized():
                return self.reply(401, {'error': 'Unauthorized'})
            if self.path == '/health':
                return self.reply(200, dict(ready=True, **backend.info))
            if self.path == '/v1/models':
                return self.reply(200, dict(object='list', data=[dict(id=m, object='model') for m in MODELS]))
            self.reply(404, {'error': 'Not found'})
        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {'error': 'Unauthorized'})
            if self.path != '/v1/chat/completions':
                return self.reply(404, {'error': 'Not found'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 131072:
                    raise ValueError('Request body must be 1..131072 bytes')
                request = prepare_request(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError, KeyError, UnicodeError):
                return self.reply(400, {'error': 'Invalid or unsupported request'})
            if not gate.acquire(blocking=False):
                return self.reply(429, {'error': 'Single GPU is busy; retry later'})
            try:
                result = backend.generate(request)
                self.reply(200, result)
                print(json.dumps(dict(event='request_complete', model=request['model'],
                    usage=result['usage'], metrics=result['metrics'])), flush=True)
            except ValueError as error:
                self.reply(422, {'error': str(error)})
            except Exception as error:
                print(json.dumps(dict(event='generation_error', type=type(error).__name__)), flush=True)
                self.reply(500, {'error': 'Generation failed; check server log'})
            finally:
                gate.release()
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/root/autodl-tmp/qwen3-30b-a3b-bf16')
    parser.add_argument('--source-run', default='/root/autodl-tmp/deeplearning-lab/rl/qwen3_moe_grpo/runs/qwen3-round2-20261022-194758-1550')
    # Longer contexts are opt-in for the verbatim C-Agent prompts, GPU validation pending.
    parser.add_argument('--max-context', type=int, choices=[2048, 4096, 8192, 16384, 32768], default=4096)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    api_key = os.environ.get('QWEN3_API_KEY', '')
    if len(api_key) < 24:
        parser.error('Set QWEN3_API_KEY to a random secret of at least 24 characters')
    backend = Backend(args.model_path, args.source_run, args.max_context)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(backend, api_key))
    print(json.dumps(dict(event='ready', **backend.info)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
