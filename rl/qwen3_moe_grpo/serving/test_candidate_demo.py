"""CPU-only boundary and replay tests. Scripted replies do not measure Qwen quality."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from candidate_cases import CASES, KNOWN
from candidate_demo import check_case, preflight, run_case
from candidate_mock import MockBusiness, TOOLS
from candidate_prompts import FILES, PromptBundle, snapshot
from protocol import parse_output, prepare_request
from agent_server import CandidateAgent


class Bundle:
    sha256 = 'scripted-test-bundle'
    def skill(self, name, industry):
        return f'{industry}:{name}:完整策略占位（仅单元测试）'
    def system(self, context):
        return json.dumps(context, ensure_ascii=False)


def response(name=None, args=None, text='这是测试最终回答。'):
    raw = ('<tool_call>' + json.dumps(dict(name=name, arguments=args), ensure_ascii=False) + '</tool_call>') if name else text
    message, reason = parse_output(raw, TOOLS)
    return dict(choices=[dict(message=message, finish_reason=reason)])


class CandidateTests(unittest.TestCase):
    def test_agent_retains_profile_and_current_job_between_turns(self):
        replies = iter([
            response('candidate_profile_tools', dict(action='update_profile', age='35', city='济南', category='保洁')),
            response('read_skill', dict(skill_name='job-service')),
            response('job_tools', dict(action='search_jobs')),
            response('job_tools', dict(action='select_active_job', selectedJobId='HC101')),
            response(text='槐荫区社区日间保洁，税前4500元一个月。'),
            response('job_tools', dict(action='job_qa', userMessage='工资到手多少？')),
            response(text='4500元是税前工资，到手金额还需确认。')])
        agent = CandidateAgent(Bundle(), 'qwen3-grpo', lambda body: next(replies))
        self.assertTrue(agent.turn('我35岁，在济南找保洁。')['completed'])
        self.assertEqual(agent.business.exposed, 'HC101')
        result = agent.turn('工资到手多少？')
        self.assertTrue(result['completed'])
        self.assertEqual(agent.business.profile['age'], '35')
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(sum(m['role'] == 'user' for m in agent.messages), 2)

    def test_agent_consent_is_narrow_and_reversible(self):
        agent = CandidateAgent(Bundle(), 'qwen3-base', None)
        agent.business = MockBusiness(Bundle(), KNOWN)
        for text in ['这个岗位可以', '不拉群', '可以拉群，但不能住家', '你说帮我拉群是什么意思？']:
            agent._consent(text)
            self.assertEqual(agent.business.consent_job, '')
        agent._consent('可以，帮我拉群。')
        self.assertEqual(agent.business.consent_job, 'HC101')
        agent._consent('暂时别拉群')
        self.assertEqual(agent.business.consent_job, '')

    def test_agent_failed_turn_cannot_replay_incomplete_protocol(self):
        agent = CandidateAgent(Bundle(), 'qwen3-base', lambda _: response('job_tools', dict(action='search_jobs')))
        self.assertFalse(agent.turn('找工作')['completed'])
        with self.assertRaises(ValueError):
            agent.turn('继续')

    def test_verbatim_snapshot_integrity_and_composition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in FILES:
                path = root / 'source' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('CONTENT:' + name)
            snapshot(root / 'source', root / 'snapshot')
            bundle = PromptBundle(root / 'snapshot')
            system = bundle.system(dict(industryCode='homecare'))
            self.assertLess(system.index('common/IDENTITY.md'), system.index('homecare/IDENTITY.md'))
            self.assertLess(system.index('homecare/IDENTITY.md'), system.index('common/AGENTS.md'))
            self.assertNotIn('general-worker/AGENTS.md', system)
            self.assertIn('common/skills/job-service/SKILL.md', bundle.skill('job-service', 'homecare'))
            (root / 'snapshot/common/TOOLS.md').write_text('tampered')
            with self.assertRaises(ValueError):
                PromptBundle(root / 'snapshot')

    def test_profile_update_affects_search_and_case_isolation(self):
        business = MockBusiness(Bundle())
        self.assertEqual(business.execute('job_tools', dict(action='search_jobs', city='济南', category='保洁'))['code'], 'SKILL_REQUIRED')
        business.execute('read_skill', dict(skill_name='job-service'))
        business.execute('candidate_profile_tools', dict(action='update_profile', age='35', city='济南', category='保洁'))
        result = business.execute('job_tools', dict(action='search_jobs'))
        self.assertEqual(result['candidateJobs'][0]['eligibility']['status'], 'MATCH')
        self.assertEqual(MockBusiness(Bundle()).profile, {})
        business.execute('candidate_profile_tools', dict(action='update_profile', age='15'))
        self.assertEqual(business.execute('job_tools', dict(action='search_jobs'))['candidateJobs'], [])

    def test_unknown_is_not_match_or_conflict(self):
        business = MockBusiness(Bundle(), dict(active='HC102'))
        result = business.job('HC102')['eligibility']
        self.assertEqual(result['status'], 'UNKNOWN')
        business.execute('candidate_profile_tools', dict(action='update_profile', qualificationsJson='{"canLiveIn":false}'))
        self.assertEqual(business.job('HC102')['eligibility']['status'], 'CONFLICT')

    def test_unknown_identity_scope_and_malformed_args(self):
        business = MockBusiness(Bundle())
        for name, args in [
            ('job_tools', dict(action='search_jobs', sessionId='other')),
            ('candidate_profile_tools', dict(action='update_profile', age=35)),
            ('candidate_profile_tools', dict(action='update_profile', qualificationsJson='[]')),
            ('candidate_profile_tools', dict(action='update_profile', qualificationsJson='{"consent":true}')),
        ]:
            with self.subTest(args=args):
                self.assertEqual(business.execute(name, args)['code'], 'INVALID_ARGUMENTS')
        self.assertEqual(business.execute('send_real_message', {})['code'], 'UNKNOWN_TOOL')

    def test_selection_requires_search_receipt(self):
        business = MockBusiness(Bundle(), dict(profile=dict(city='济南', category='保洁', age='35')))
        business.execute('read_skill', dict(skill_name='job-service'))
        self.assertEqual(business.execute('job_tools', dict(action='select_active_job', selectedJobId='HC101'))['status'], 'ERROR')
        business.execute('job_tools', dict(action='search_jobs'))
        self.assertEqual(business.execute('job_tools', dict(action='select_active_job', selectedJobId='HC101'))['status'], 'SUCCESS')
        self.assertEqual(business.exposed, '')

    def test_acceptance_cannot_grant_group_consent(self):
        business = MockBusiness(Bundle(), KNOWN)
        for name in ('job-service', 'group-delivery'):
            business.execute('read_skill', dict(skill_name=name))
        business.execute('job_tools', dict(action='record_job_feedback', feedback='accepted'))
        self.assertEqual(business.execute('group_delivery_tools', dict(action='create_or_reuse_group'))['code'], 'CONSENT_OR_ACTIVE_JOB_REQUIRED')
        self.assertFalse(business.group)

    def test_group_success_idempotency_and_binding(self):
        business = MockBusiness(Bundle(), dict(**KNOWN, consent_job='HC101'))
        for name in ('job-service', 'group-delivery'):
            business.execute('read_skill', dict(skill_name=name))
        self.assertEqual(business.execute('group_delivery_tools', dict(action='create_or_reuse_group'))['code'], 'READY_CHECK_REQUIRED')
        business.execute('job_tools', dict(action='get_merchant_delivery_context'))
        self.assertTrue(business.execute('group_delivery_tools', dict(action='check_group_delivery_ready'))['ready'])
        first = business.execute('group_delivery_tools', dict(action='create_or_reuse_group'))
        second = business.execute('group_delivery_tools', dict(action='create_or_reuse_group'))
        self.assertEqual(first['groupId'], second['groupId'])
        self.assertTrue(second['duplicate'])
        self.assertEqual(business.execute('group_delivery_tools', dict(action='send_private_grouped_notice', groupId='wrong'))['code'], 'GROUP_MISMATCH')
        self.assertTrue(business.execute('group_delivery_tools', dict(action='send_private_grouped_notice', groupId=first['groupId']))['sent'])

    def test_failure_does_not_create_receipt(self):
        business = MockBusiness(Bundle(), dict(**KNOWN, consent_job='HC101', fail='group_delivery_tools.create_or_reuse_group'))
        for name in ('job-service', 'group-delivery'):
            business.execute('read_skill', dict(skill_name=name))
        business.execute('job_tools', dict(action='get_merchant_delivery_context'))
        business.execute('group_delivery_tools', dict(action='check_group_delivery_ready'))
        self.assertEqual(business.execute('group_delivery_tools', dict(action='create_or_reuse_group'))['status'], 'ERROR')
        self.assertFalse(business.group)
        self.assertEqual(business.execute('group_delivery_tools', dict(action='send_private_grouped_notice', groupId='invented'))['status'], 'ERROR')

    def test_profile_change_invalidates_ready(self):
        business = MockBusiness(Bundle(), dict(**KNOWN, consent_job='HC101'))
        for name in ('job-service', 'group-delivery'):
            business.execute('read_skill', dict(skill_name=name))
        business.execute('job_tools', dict(action='get_merchant_delivery_context'))
        business.execute('group_delivery_tools', dict(action='check_group_delivery_ready'))
        business.execute('candidate_profile_tools', dict(action='update_profile', age='15'))
        self.assertEqual(business.execute('group_delivery_tools', dict(action='create_or_reuse_group'))['code'], 'JOB_REQUIREMENTS_NOT_MET')

    def test_switch_clears_old_scope_and_preserves_name(self):
        business = MockBusiness(Bundle(), dict(**KNOWN, consent_job='HC101'))
        business.execute('read_skill', dict(skill_name='job-service'))
        business.execute('candidate_session_state_tools', dict(action='update_primary_industry', primaryIndustry='general-worker'))
        self.assertEqual(business.name, '李老师')
        self.assertFalse(business.active or business.loaded or business.consent_job)
        self.assertNotIn('category', business.profile)
        self.assertEqual(business.execute('job_tools', dict(action='search_jobs', city='济南', category='普工'))['code'], 'SKILL_REQUIRED')

    def test_memory_does_not_update_profile(self):
        business = MockBusiness(Bundle())
        business.execute('memory_write', dict(category='profile', fact='今年35岁'))
        self.assertEqual(business.profile, {})
        rows = business.execute('memory_search', dict(query='年龄'))['matches']
        self.assertEqual(business.execute('memory_get', dict(path=rows[0]['path']))['matches'], rows)

    def test_runner_reinjects_context_and_roundtrips_tool_results(self):
        replies = iter([
            response('candidate_profile_tools', dict(action='update_profile', age='35', city='济南', category='保洁')),
            response('read_skill', dict(skill_name='job-service')),
            response('job_tools', dict(action='search_jobs')),
            response('job_tools', dict(action='select_active_job', selectedJobId='HC101')),
            response(text='槐荫区社区日间保洁是白班，税前4500元一个月。'),
        ])
        requests = []
        def send(body):
            prepare_request(body)
            requests.append(copy.deepcopy(body))
            return next(replies)
        result = run_case(CASES[2], 'qwen3-base', send, Bundle())
        self.assertTrue(result['completed'])
        self.assertEqual(json.loads(requests[1]['messages'][0]['content'])['profile']['age'], '35')
        self.assertEqual(result['checks']['missing_expected_actions'], [])
        self.assertTrue(result['checks']['needs_human_review'])

    def test_runner_removes_old_skill_pair_after_switch(self):
        replies = iter([
            response('read_skill', dict(skill_name='job-service')),
            response('candidate_session_state_tools', dict(action='update_primary_industry', primaryIndustry='general-worker')),
            response('read_skill', dict(skill_name='job-service')),
            response(),
        ])
        requests = []
        def send(body):
            prepare_request(body)
            requests.append(copy.deepcopy(body))
            return next(replies)
        self.assertTrue(run_case(CASES[13], 'qwen3-grpo', send, Bundle())['completed'])
        self.assertNotIn('homecare:job-service', json.dumps(requests[2], ensure_ascii=False))
        self.assertIn('general-worker:job-service', json.dumps(requests[3], ensure_ascii=False))

    def test_loop_stops_and_guard_does_not_mask_model_attempt(self):
        result = run_case(CASES[9], 'qwen3-base', lambda _: response('group_delivery_tools', dict(action='create_or_reuse_group')), Bundle())
        self.assertFalse(result['completed'])
        self.assertIn('duplicate', result['error'])
        self.assertTrue(result['checks']['forbidden_attempts'])

    def test_preflight_is_full_template_and_reports_overflow(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.tools = kwargs['tools']
                return json.dumps(messages, ensure_ascii=False)
            def encode(self, text, **kwargs):
                return list(text)  # Deliberately fake tokenizer for wiring only.
        tokenizer = Tokenizer()
        report = preflight(Bundle(), tokenizer, max_context=32)
        self.assertFalse(report['fits'])
        self.assertFalse(report['gpu_memory_validated'])
        self.assertEqual(len(report['cases']), len(CASES))
        self.assertEqual(len(tokenizer.tools), 8)


if __name__ == '__main__':
    unittest.main()
