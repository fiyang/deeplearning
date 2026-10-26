"""Small, non-streaming chat API subset for the local Qwen3 experiment."""
import copy
import json
import math
import re
import uuid

MODELS = ('qwen3-base', 'qwen3-grpo')


def service_seconds(budget, price, started, platform, now):
    if not all(math.isfinite(x) for x in [budget, price, started, platform, now]) or budget <= 0 or price <= 0:
        raise ValueError('Invalid budget/price')
    if started > now or platform <= now or platform > started + 5400:
        raise ValueError('Platform shutdown must be within 90 minutes of boot')
    deadline = min(started + budget / price * 3600 - 180, platform - 120, started + 3600)
    seconds = int(deadline - now)
    if seconds < 300:
        raise ValueError('Insufficient budget/time remaining')
    return seconds


def prepare_request(body):
    if not isinstance(body, dict):
        raise ValueError('Request must be an object')
    allowed = {'model', 'messages', 'tools', 'tool_choice', 'max_tokens', 'temperature', 'top_p', 'stream'}
    if set(body) - allowed:
        raise ValueError('Unsupported request fields: ' + ', '.join(sorted(set(body) - allowed)))
    if body.get('model') not in MODELS:
        raise ValueError('model must be qwen3-base or qwen3-grpo')
    if body.get('stream', False) is not False:
        raise ValueError('Streaming is not implemented in this learning server')
    choice = body.get('tool_choice', 'auto')
    if choice not in ('auto', 'none'):
        raise ValueError('Only auto and none tool_choice are supported')
    max_tokens = body.get('max_tokens', 512)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
        raise ValueError('max_tokens must be an integer in [1, 1024]')
    temperature, top_p = body.get('temperature', 0.7), body.get('top_p', 0.8)
    if type(temperature) not in (int, float) or not 0 <= temperature <= 1.5:
        raise ValueError('temperature must be in [0, 1.5]')
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')
    tools = copy.deepcopy(body.get('tools', []))
    if not isinstance(tools, list) or len(tools) > 8:
        raise ValueError('At most eight tool schemas are supported')
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function':
            raise ValueError('Only function tools are supported')
        fn = tool.get('function', {})
        name = fn.get('name') if isinstance(fn, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,63}', name):
            raise ValueError('Invalid function name')
        if name in names or not isinstance(fn.get('parameters'), dict):
            raise ValueError('Duplicate function or missing parameter schema')
        names.add(name)
    messages = copy.deepcopy(body.get('messages'))
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise ValueError('messages must contain 1..32 messages')
    pending, seen = set(), set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError('Invalid message')
        role = message.get('role')
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Unsupported role')
        if role == 'system' and index != 0:
            raise ValueError('System message is allowed only at the beginning')
        if message.get('content') is not None and not isinstance(message['content'], str):
            raise ValueError('Only text content is supported')
        calls = message.get('tool_calls', [])
        if not isinstance(calls, list) or len(calls) > 8:
            raise ValueError('Invalid historical tool_calls')
        if role != 'assistant' and not isinstance(message.get('content'), str):
            raise ValueError('User/system/tool messages must contain text')
        if calls and role != 'assistant':
            raise ValueError('Only assistant messages may contain tool_calls')
        if pending and role != 'tool':
            raise ValueError('Every tool call needs a result before the next message')
        if role == 'tool':
            ident = message.get('tool_call_id')
            if ident not in pending:
                raise ValueError('Unmatched tool result')
            pending.remove(ident)
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                raise ValueError('Invalid historical function')
            fn = call['function']
            ident = call.get('id')
            if not isinstance(ident, str) or ident in seen or call.get('type') != 'function':
                raise ValueError('Invalid tool call identity')
            seen.add(ident)
            pending.add(ident)
            args = fn.get('arguments')
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict) or not isinstance(fn.get('name'), str):
                raise ValueError('Invalid historical tool call')
            # The Qwen template expects an object, the HTTP API uses JSON strings.
            fn['arguments'] = args
    if pending or messages[-1]['role'] not in ('user', 'tool'):
        raise ValueError('Request must end with a user message or completed tool results')
    return dict(model=body['model'], messages=messages, tools=tools if choice == 'auto' else [],
                max_tokens=max_tokens, temperature=temperature, top_p=top_p)


def parse_output(text, tools, truncated=False):
    """Only completed Hermes calls outside reasoning can be executed by a client."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if '<think>' in text or '</think>' in text:
        raise ValueError('Unbalanced reasoning tags')
    matches = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S))
    remainder = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S).strip()
    if '<tool_call' in remainder or '</tool_call' in remainder or (truncated and matches):
        raise ValueError('Incomplete/truncated tool response; no calls executed')
    allowed = {t['function']['name'] for t in tools}
    calls = []
    for match in matches:
        item = json.loads(match.group(1))
        if not isinstance(item, dict) or set(item) != {'name', 'arguments'}:
            raise ValueError('Malformed tool call')
        if not isinstance(item['name'], str) or item['name'] not in allowed or not isinstance(item['arguments'], dict):
            raise ValueError('Unknown tool or non-object arguments')
        calls.append(dict(id='call_' + uuid.uuid4().hex[:16], type='function',
                          function=dict(name=item['name'], arguments=json.dumps(item['arguments'], ensure_ascii=False))))
    if len(calls) > 8:
        raise ValueError('Too many calls in one response')
    if not calls and not remainder:
        raise ValueError('Empty model response')
    message = dict(role='assistant', content=remainder or None)
    if calls:
        message['tool_calls'] = calls
    return message, ('tool_calls' if calls else 'length' if truncated else 'stop')
