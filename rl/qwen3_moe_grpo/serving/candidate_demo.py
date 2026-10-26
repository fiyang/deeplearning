"""Bounded candidate-facing A/B experiment using verbatim business prompts."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

from candidate_cases import CASES
from candidate_mock import MockBusiness, TOOLS
from candidate_prompts import PromptBundle, SKILLS
from protocol import prepare_request
from recruiting_demo import chat


def remove_loaded_skills(messages):
    # Called only after an industry switch. Remove complete old skill pairs.
    ids = {c['id'] for m in messages for c in m.get('tool_calls', [])
           if c['function']['name'] == 'read_skill'}
    cleaned = []
    for message in messages:
        if message.get('tool_call_id') in ids:
            continue
        if message.get('tool_calls'):
            kept = [c for c in message['tool_calls'] if c['id'] not in ids]
            if not kept and not message.get('content'):
                continue
            message = dict(message, tool_calls=kept)
        cleaned.append(message)
    return cleaned


def run_case(case, model, send, bundle, max_rounds=14, max_calls=18, *, business=None, messages=None):
    business = business if business is not None else MockBusiness(bundle, case.get('seed'))
    messages = messages if messages is not None else copy.deepcopy(case.get('history', []))
    messages.append(dict(role='user', content=case['prompt']))
    trace, calls, seen = [], 0, set()
    result = dict(case_id=case['id'], model=model, prompt=case['prompt'], history=case.get('history', []),
                  prompt_manifest_sha256=bundle.sha256, trace=trace, completed=False,
                  backend='synthetic-in-memory', optimizer_updates=0)
    started = time.perf_counter()
    try:
        for _ in range(max_rounds):
            system = bundle.system(business.context())
            body = dict(model=model, messages=[dict(role='system', content=system)] + messages,
                        tools=TOOLS, temperature=0, top_p=1, max_tokens=512)
            prepare_request(body)
            response = send(body)
            choice = response['choices'][0]
            message = choice['message']
            tool_calls = message.get('tool_calls', [])
            trace.append(dict(kind='model', message=message, finish_reason=choice['finish_reason'],
                              system_sha256=hashlib.sha256(system.encode()).hexdigest(),
                              usage=response.get('usage'), metrics=response.get('metrics')))
            if choice['finish_reason'] == 'length':
                raise ValueError('Truncated response')
            messages.append(message)
            if not tool_calls:
                if choice['finish_reason'] != 'stop' or not message.get('content'):
                    raise ValueError('Invalid final answer')
                result.update(completed=True, answer=message['content'])
                # Exposure is checked manually; do not grant consent from generated text.
                break
            if choice['finish_reason'] != 'tool_calls' or calls + len(tool_calls) > max_calls:
                raise ValueError('Tool call budget/finish reason violation')
            # Skill retrieval and dependent actions must wait for separate model rounds.
            mixed_skill = len(tool_calls) > 1 and any(c['function']['name'] == 'read_skill' for c in tool_calls)
            old_industry = business.industry
            for call in tool_calls:
                calls += 1
                fn, args = call['function'], json.loads(call['function']['arguments'])
                fingerprint = json.dumps([fn, business.context(), sorted(business.loaded), business.group], sort_keys=True)
                if fingerprint in seen:
                    raise ValueError('Unchanged duplicate tool call; stopped to prevent loop')
                seen.add(fingerprint)
                output = (business.error(fn['name'], args, 'SKILL_MUST_FINISH_FIRST') if mixed_skill
                          else business.execute(fn['name'], args))
                trace.append(dict(kind='tool', name=fn['name'], arguments=args, result=output))
                messages.append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(output, ensure_ascii=False)))
            if business.industry != old_industry:
                messages[:] = remove_loaded_skills(messages)
        if not result['completed']:
            raise ValueError('Model round budget exhausted')
    except (ValueError, TypeError, KeyError, OSError) as error:
        result['error'] = str(error)
    result.update(seconds=time.perf_counter()-started, tool_calls=calls, final_context=business.context())
    result['checks'] = check_case(case, result)
    return result


def check_case(case, result):
    events = [e for e in result['trace'] if e['kind'] == 'tool']
    actions = [e['name'] + ('.' + e['arguments']['action'] if 'action' in e['arguments'] else '') for e in events]
    missing = [name for name in case.get('expected', []) if name not in actions]
    forbidden = [name for name in actions if any(name == rule or name.startswith(rule + '.') for rule in case.get('forbid', []))]
    successful = [a for a, e in zip(actions, events) if e['result'].get('status') == 'SUCCESS']
    answer = result.get('answer', '')
    markers = [s for s in ('<think>', '<analysis>', 'job_tools', 'group_delivery_tools', 'read_skill',
                           'active_job', 'sessionId', '我应该', '我需要调用', '根据Skill') if s in answer]
    return dict(completed=result['completed'], missing_expected_actions=missing, forbidden_attempts=forbidden,
                successful_actions=successful, tool_error_count=sum(e['result'].get('status') == 'ERROR' for e in events),
                possible_internal_leaks=markers, needs_human_review=True,
                review=case.get('review', '人工检查参数、事实、授权、顺序和话术；自动检查不代表业务成功率。'))


def preflight(bundle, tokenizer, max_context=32768):
    """Exact chat-template length on CPU; loads only a LOCAL tokenizer, no weights."""
    rows = []
    for case in CASES:
        business = MockBusiness(bundle, case.get('seed'))
        messages = [dict(role='system', content=bundle.system(business.context()))] + copy.deepcopy(case.get('history', []))
        messages.append(dict(role='user', content=case['prompt']))
        def count(items):
            prepared = prepare_request(dict(model='qwen3-base', messages=items, tools=TOOLS))
            text = tokenizer.apply_chat_template(prepared['messages'], tools=TOOLS, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            return len(tokenizer.encode(text, add_special_tokens=False))
        initial = count(messages)
        for i, skill in enumerate(SKILLS):
            messages.extend([dict(role='assistant', content=None, tool_calls=[dict(id=f'preflight_{i}', type='function',
                function=dict(name='read_skill', arguments=json.dumps(dict(skill_name=skill))))]),
                dict(role='tool', tool_call_id=f'preflight_{i}', content=json.dumps(dict(
                    status='SUCCESS', skill=skill, content=bundle.skill(skill, business.industry)), ensure_ascii=False))])
        skills = count(messages)
        rows.append(dict(case_id=case['id'], initial_tokens=initial, with_both_skills_tokens=skills,
                         reserved_followup_and_output_tokens=4096, planned_context=max_context,
                         fits_with_reserve=skills + 4096 <= max_context))
    return dict(kind='cpu-tokenizer-preflight', prompt_manifest_sha256=bundle.sha256, cases=rows,
                fits=all(r['fits_with_reserve'] for r in rows), gpu_memory_validated=False,
                note='Follow-up reserve is planning headroom, not a bound; server rejects any overlong request.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-dir', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', choices=['qwen3-base', 'qwen3-grpo', 'both'], default='both')
    parser.add_argument('--case', action='append', choices=[c['id'] for c in CASES])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-tokenizer', type=Path, help='Only count tokens using an existing local model directory')
    parser.add_argument('--max-context', type=int, default=32768)
    args = parser.parse_args()
    bundle = PromptBundle(args.prompt_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preflight_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.preflight_tokenizer), local_files_only=True, trust_remote_code=False)
        report = preflight(bundle, tokenizer, args.max_context)
        with args.output.open('x') as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False))
        if not report['fits']:
            raise SystemExit('Full prompt does not fit. Do not silently truncate business instructions.')
        return
    key = os.environ.get('QWEN3_API_KEY')
    if not key:
        parser.error('Set QWEN3_API_KEY')
    models = ['qwen3-base', 'qwen3-grpo'] if args.model == 'both' else [args.model]
    cases = [c for c in CASES if not args.case or c['id'] in args.case]
    with args.output.open('x') as output:
        # Alternate model order per case to reduce a fixed warmup/order bias.
        for index, case in enumerate(cases):
            for model in models if index % 2 == 0 else list(reversed(models)):
                result = run_case(case, model, lambda body: chat(args.base_url, key, body), bundle)
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
                output.flush()
                print(json.dumps({k: v for k, v in result.items() if k not in ('trace', 'history')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
