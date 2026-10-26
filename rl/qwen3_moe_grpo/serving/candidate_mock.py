"""C-side fixture executor. No network, real candidates, messages or persistent writes.

Business action names/fields follow ConversationDemo*Tools. This is an explicit
subset, NOT execution of Java tools or a production-equivalence implementation.
Consent is fixture-owned evidence, never a model-supplied boolean.
"""
import copy
import json

from candidate_prompts import INDUSTRIES, SKILLS
from recruiting_demo import tool


def schema(name, description, fields, required=('action',), enums=None):
    properties = {field: dict(type='string') for field in fields.split()}
    for field, values in (enums or {}).items():
        properties[field]['enum'] = values
    return tool(name, description, properties, list(required))


PROFILE_FIELDS = ('city category age gender salaryExpectation preferences factoryExperience '
                  'shiftPreference accommodationRequired arrivalTime rejectConditions qualificationsJson')
JOB_ACTIONS = ['get_current_job_context', 'job_qa', 'search_jobs', 'select_active_job',
               'record_job_feedback', 'get_interview_slots', 'record_interview_appointment',
               'get_merchant_delivery_context']
GROUP_ACTIONS = ['check_group_delivery_ready', 'create_or_reuse_group', 'send_group_welcome',
                 'send_private_grouped_notice', 'schedule_group_activation']
TOOLS = [
    schema('read_skill', '读取当前行业的完整业务策略。与业务动作分轮调用，先等待读取结果。',
           'skill_name', ('skill_name',), {'skill_name': list(SKILLS)}),
    schema('candidate_profile_tools', '读取或更新当前实际求职者画像；只写用户明确表达的字段。年龄为字符串。'
           'qualificationsJson为JSON字符串，支持experienceYears、experienceCategory、certificates、canLiveIn。',
           'action ' + PROFILE_FIELDS, enums={'action': ['get_profile', 'update_profile']}),
    schema('candidate_session_state_tools', '权威行业状态；明确切行业后先更新，再读新行业策略并处理岗位。',
           'action primaryIndustry reason', enums={'action': ['get_state', 'update_primary_industry']}),
    schema('job_tools', '岗位一体化工具。字段均可省略未知值。search_jobs使用画像或显式条件；'
           '选岗使用selectedJobId；答疑使用userMessage。contextJson支持district、excludeJobIds、liveInRequired。'
           '所有具体岗位事实以成功结果为准。身份作用域由应用注入。',
           'action currentJobId userMessage city category age gender salaryExpectation preferences feedback '
           'rejectReason selectedJobId interviewTime interviewAddress contextJson', enums={'action': JOB_ACTIONS}),
    schema('group_delivery_tools', '仅当前岗位获得用户明确授权后使用。先获取商家上下文、检查ready，再创建或复用，'
           '之后用回执groupId发送欢迎和私聊告知。所有动作仅在测试内存中模拟。',
           'action jobId merchantId groupId interviewTime interviewAddress deliveryContextJson',
           enums={'action': GROUP_ACTIONS}),
    schema('memory_search', '查询当前会话历史偏好，不是岗位事实或当前交付状态。', 'query', ('query',)),
    schema('memory_get', '按memory_search返回的路径读取事实。', 'path', ('path',)),
    schema('memory_write', '记录用户明确的稳定事实。当前画像要先更新；不写当前建群状态。fact不超过300字。',
           'category fact', ('category', 'fact')),
]

# Deliberately fabricated, small fixtures. Missing benefits stay unknown.
JOBS = {
    'HC101': dict(jobId='HC101', industry='homecare', jobName='社区日间保洁', category='保洁', city='济南',
                  district='槐荫区', salary='4500元/月（税前）', address='槐荫区示例社区A',
                  shift='白班', liveInRequired=False, merchantId='MOCK-M1',
                  hardRequirements=dict(minAge=18), interviewSlots=[]),
    'HC102': dict(jobId='HC102', industry='homecare', jobName='家庭住家保姆', category='保姆', city='济南',
                  district='历下区', salary='6500元/月（税前）', address='历下区示例社区B',
                  liveInRequired=True, merchantId='MOCK-M2',
                  hardRequirements=dict(minAge=18, canLiveIn=True, experienceYears=1, experienceCategory='保姆'),
                  interviewSlots=[]),
    'GW101': dict(jobId='GW101', industry='general-worker', jobName='白班包装普工', category='普工', city='济南',
                  district='槐荫区', salary='综合工资5000至6000元/月（非到手承诺）', address='槐荫区示例厂区C',
                  shift='白班', merchantId='MOCK-M3', hardRequirements=dict(minAge=18), interviewSlots=[]),
}


class MockBusiness:
    def __init__(self, bundle, seed=None):
        seed = copy.deepcopy(seed or {})
        self.bundle = bundle
        self.industry = seed.get('industry', 'homecare')
        self.profile = seed.get('profile', {})
        self.active = seed.get('active', '')
        self.exposed = seed.get('exposed', self.active)
        self.name = seed.get('name', '李老师')
        # Supplied by the human-labelled test fixture, bound to this exact job.
        self.consent_job = seed.get('consent_job', '')
        self.fail = seed.get('fail', '')
        self.memories = seed.get('memories', [])
        self.loaded, self.candidates, self.rejected = set(), set(), set()
        self.group, self.merchant_checked, self.ready_job = {}, '', ''
        self.appointment = {}

    def context(self):
        return dict(industryCode=self.industry, currentName=self.name, profile=copy.deepcopy(self.profile),
                    active_job_id=self.active, recently_exposed_job_id=self.exposed,
                    actualCandidate='当前求职者（若代亲友求职则以对话中实际求职者为准）',
                    evaluation_only=False)

    def result(self, name, args, **values):
        return dict(tool=name, action=args.get('action', name), status='SUCCESS', simulated=True, **values)

    def error(self, name, args, code, **values):
        result = self.result(name, args, **values)
        result.update(status='ERROR', code=code)
        return result

    def eligibility(self, job, profile=None):
        profile = self.profile if profile is None else profile
        missing, conflicts = [], []
        requirements = job.get('hardRequirements', {})
        for field, required in requirements.items():
            key = 'age' if field == 'minAge' else field
            value = profile.get(key)
            if value is None or value == '':
                missing.append(key)
            elif field in ('minAge', 'experienceYears'):
                if float(value) < required:
                    conflicts.append(key)
            elif value != required:
                conflicts.append(key)
        if profile.get('city') and profile['city'] != job['city']:
            conflicts.append('city')
        if profile.get('canLiveIn') is False and job.get('liveInRequired'):
            conflicts.append('canLiveIn')
        if profile.get('shiftPreference') == '白班' and job.get('shift') != '白班':
            conflicts.append('shiftPreference')
        return dict(status='CONFLICT' if conflicts else 'UNKNOWN' if missing else 'MATCH',
                    missingCandidateFields=missing, conflicts=conflicts, scope='declared fixture requirements only')

    def job(self, ident):
        job = JOBS.get(ident, {})
        if job.get('industry') != self.industry:
            return {}
        return dict(copy.deepcopy(job), eligibility=self.eligibility(job))

    def execute(self, name, args):
        definitions = {t['function']['name']: t['function']['parameters'] for t in TOOLS}
        if name not in definitions:
            return self.error(name, {}, 'UNKNOWN_TOOL')
        spec = definitions[name]
        if (not isinstance(args, dict) or set(args) - set(spec['properties'])
                or not set(spec['required']) <= set(args)):
            return self.error(name, {}, 'INVALID_ARGUMENTS')
        for key, value in args.items():
            prop = spec['properties'][key]
            if (not isinstance(value, str) or not value.strip() or len(value) > 2000
                    or 'enum' in prop and value not in prop['enum']):
                return self.error(name, args, 'INVALID_ARGUMENTS')
        action = args.get('action', name)
        if self.fail == name + '.' + action:
            return self.error(name, args, 'SIMULATED_FAILURE')
        if name == 'read_skill':
            skill = args['skill_name']
            self.loaded.add(skill)
            return self.result(name, args, skill=skill, industry=self.industry,
                               content=self.bundle.skill(skill, self.industry))
        if name == 'candidate_profile_tools':
            if action == 'update_profile':
                update = {k: v for k, v in args.items() if k != 'action'}
                try:
                    if 'age' in update and (not update['age'].isdigit() or not 0 <= int(update['age']) <= 120):
                        raise ValueError('Invalid age')
                    qualification = json.loads(update.pop('qualificationsJson', '{}'))
                    if not isinstance(qualification, dict):
                        raise ValueError('Qualification must be object')
                    for key, value in qualification.items():
                        valid = ((key == 'canLiveIn' and type(value) is bool)
                                 or (key == 'experienceYears' and type(value) in (int, float) and 0 <= value <= 100)
                                 or (key == 'experienceCategory' and isinstance(value, str) and bool(value))
                                 or (key == 'certificates' and isinstance(value, list)
                                     and all(isinstance(v, str) and v for v in value)))
                        if not valid:
                            raise ValueError('Invalid qualification')
                    update.update(qualification)
                except (ValueError, TypeError):
                    return self.error(name, args, 'INVALID_ARGUMENTS')
                self.profile.update(update)
                self.ready_job = ''
            return self.result(name, args, profile=copy.deepcopy(self.profile))
        if name == 'candidate_session_state_tools':
            if action == 'update_primary_industry':
                industry = args.get('primaryIndustry')
                if industry not in INDUSTRIES:
                    return self.error(name, args, 'UNSUPPORTED_INDUSTRY', supportedIndustries=list(INDUSTRIES))
                if industry != self.industry:
                    self.industry = industry
                    self.active = self.exposed = self.consent_job = self.ready_job = self.merchant_checked = ''
                    self.loaded.clear()
                    self.candidates.clear()
                    self.group, self.appointment = {}, {}
                    self.profile = {k: v for k, v in self.profile.items() if k in ('city', 'age', 'gender', 'salaryExpectation')}
            return self.result(name, args, primaryIndustry=self.industry, supportedIndustries=list(INDUSTRIES))
        if name.startswith('memory_'):
            if name == 'memory_write':
                if len(args['fact']) > 300:
                    return self.error(name, args, 'INVALID_ARGUMENTS')
                if args['fact'] not in self.memories:
                    self.memories.append(args['fact'])
                return self.result(name, args, saved=True)
            rows = [dict(path=f'memory://fixture/{i}', fact=fact) for i, fact in enumerate(self.memories)]
            if name == 'memory_search':
                # Small fixture recall, not production search/ranking.
                return self.result(name, args, matches=rows)
            rows = [r for r in rows if r['path'] == args['path']]
            return self.result(name, args, matches=rows)
        required_skill = 'job-service' if name == 'job_tools' else 'group-delivery'
        if required_skill not in self.loaded:
            return self.error(name, args, 'SKILL_REQUIRED')
        if name == 'job_tools':
            return self._job_action(args)
        return self._group_action(args)

    def _job_action(self, args):
        name, action = 'job_tools', args['action']
        if action == 'search_jobs':
            city, category = (args.get(k, self.profile.get(k, '')) for k in ('city', 'category'))
            if not city or not category:
                return self.error(name, args, 'MISSING_REQUIREMENTS')
            try:
                constraints = json.loads(args.get('contextJson', '{}'))
                if not isinstance(constraints, dict) or set(constraints) - {'district', 'excludeJobIds', 'liveInRequired'}:
                    raise ValueError('Unsupported constraints')
                if ('district' in constraints and not isinstance(constraints['district'], str)
                        or 'excludeJobIds' in constraints and (not isinstance(constraints['excludeJobIds'], list)
                            or not all(isinstance(v, str) for v in constraints['excludeJobIds']))
                        or 'liveInRequired' in constraints and type(constraints['liveInRequired']) is not bool):
                    raise ValueError('Invalid constraint')
            except (ValueError, TypeError):
                return self.error(name, args, 'INVALID_ARGUMENTS')
            excluded = self.rejected | set(constraints.get('excludeJobIds', []))
            rows = [self.job(k) for k, job in JOBS.items() if job['industry'] == self.industry
                    and job['city'] == city and (category in job['category'] or category in job['jobName'])
                    and k not in excluded and constraints.get('district', '').replace('附近', '') in job['district']
                    and ('liveInRequired' not in constraints or job.get('liveInRequired') == constraints['liveInRequired'])]
            rows = [r for r in rows if r['eligibility']['status'] != 'CONFLICT']
            self.candidates = {r['jobId'] for r in rows}
            return self.result(name, args, candidateJobs=rows, normalized=True,
                               nextStep='select_active_job' if rows else 'collect_requirements')
        if action == 'select_active_job':
            ident = args.get('selectedJobId', '')
            if ident not in self.candidates or ident in self.rejected or not self.job(ident):
                return self.error(name, args, 'JOB_NOT_IN_CURRENT_RESULTS')
            if self.job(ident)['eligibility']['status'] == 'CONFLICT':
                return self.error(name, args, 'JOB_REQUIREMENTS_NOT_MET')
            if self.active != ident:
                self.consent_job = self.ready_job = self.merchant_checked = ''
                self.group, self.appointment = {}, {}
            self.active = ident
            # Selection is not exposure: runner records exposure only after a final reply.
            return self.result(name, args, activeJob=self.job(ident), exposureRequired=True)
        ident = args.get('currentJobId', self.active)
        if ident != self.active or not self.job(ident):
            return self.error(name, args, 'NO_ACTIVE_JOB')
        job = self.job(ident)
        if action in ('get_current_job_context', 'job_qa'):
            return self.result(name, args, activeJob=job, hasActiveJob=True,
                               forbiddenInferences=['未返回福利不能补写', '税前或综合工资不能承诺到手', '未返回面试进展不能编造'])
        if action == 'record_job_feedback':
            if args.get('feedback') not in ('accepted', 'rejected', 'hesitant', 'asked_detail'):
                return self.error(name, args, 'INVALID_ARGUMENTS')
            if args['feedback'] == 'rejected':
                self.rejected.add(ident)
                self.consent_job = self.ready_job = ''
            return self.result(name, args, feedback=args['feedback'], activeJob=job)
        if action == 'get_interview_slots':
            return self.result(name, args, availableSlots=job['interviewSlots'], interviewAddress=job['address'])
        if action == 'record_interview_appointment':
            return self.error(name, args, 'NO_AVAILABLE_SLOTS')
        self.merchant_checked = ident
        return self.result(name, args, deliveryContext=dict(jobId=ident, merchantId=job['merchantId'], industry=self.industry))

    def _group_action(self, args):
        name, action = 'group_delivery_tools', args['action']
        ident = args.get('jobId', self.active)
        if not ident or ident != self.active or ident != self.exposed or ident != self.consent_job or ident in self.rejected:
            return self.error(name, args, 'CONSENT_OR_ACTIVE_JOB_REQUIRED')
        job = self.job(ident)
        if not job or job['eligibility']['status'] != 'MATCH':
            return self.error(name, args, 'JOB_REQUIREMENTS_NOT_MET')
        if args.get('merchantId', job['merchantId']) != job['merchantId']:
            return self.error(name, args, 'MERCHANT_MISMATCH')
        if action == 'check_group_delivery_ready':
            ready = self.merchant_checked == ident
            self.ready_job = ident if ready else ''
            return self.result(name, args, ready=ready, existingGroup=copy.deepcopy(self.group))
        if self.ready_job != ident:
            return self.error(name, args, 'READY_CHECK_REQUIRED')
        if action == 'create_or_reuse_group':
            duplicate = bool(self.group)
            if not self.group:
                self.group = dict(groupId='MOCK-GROUP-' + ident, jobId=ident, groupStatus='SUCCESS',
                                  groupName='测试求职对接群', welcomeSent=False)
            return self.result(name, args, **self.group, duplicate=duplicate,
                               userFacingHint='本次模拟对接流程已完成，未创建真实微信群。')
        if not self.group or args.get('groupId') != self.group['groupId']:
            return self.error(name, args, 'GROUP_MISMATCH')
        if action == 'send_group_welcome':
            self.group['welcomeSent'] = True
        if action == 'schedule_group_activation':
            return self.result(name, args, scheduled=True, simulatedOnly=True)
        return self.result(name, args, sent=True, userFacingMessage='本次模拟对接流程已完成，未创建真实微信群。')
