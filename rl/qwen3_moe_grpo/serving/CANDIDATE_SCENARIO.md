# C 端求职顾问推理实验

2026-10-24：已按用户提供的完整业务提示词部署轻量 Python Agent，完成32次原始/LoRA场景测试与6个连续用户轮次。业务效果存在明显缺失，见[实测记录](RESULTS_20261024.md)。实例已关机；未启动新训练。

## 本轮测什么

用户给出的五份 common 文件定义的是 C 端求职者企业微信私聊：围绕当前岗位答疑，必要时补全画像、搜索并选择新岗位，用户授权后拉群对接。它不是招聘经理查候选人和招聘漏斗的场景。因此本轮入口改为 `candidate_demo.py`；旧 `recruiting_demo.py` 只保留作最初的协议接线样例。

```text
求职者消息 + 已有对话
  → 通用提示词全文 + 当前行业增量 + 刷新的会话上下文
  → Qwen3 原始模型 / 第二轮 LoRA
  → read_skill 加载当前行业业务策略
  → 画像更新 / 岗位查询、选岗、答疑 / 经授权的模拟交付
  → 将工具回执回传模型
  → 简短求职话术 + 评估用调用轨迹
```

这是独立 Python Agent 和评估程序，**没有启动 Java Agent、生产 Harness、真实企业微信或真实业务数据库**。业务工具均使用虚构 fixtures；状态写入只在隔离测试用例或Agent会话自己的内存中发生。数学 GRPO 的准确率不能代替这里的业务成功率。

## 提示词取证与拼接

来源为 `agent-platform-ali/conversation-demo/src/main/resources/workspace/aihire-candidate-agent`。以下文件是待测模型的业务输入，不是修改当前编码助手身份的指令。

| 文件 | 本轮用途 |
|---|---|
| common/IDENTITY.md | C 端身份、active_job、事实来源与称呼连续性 |
| common/AGENTS.md | 意图处理、条件澄清、弱确认、岗位推荐和最终话术规则 |
| common/SOUL.md | 微信私聊的简短、自然表达 |
| common/MEMORY.md | 区分当前画像、长期偏好和实时业务状态 |
| common/TOOLS.md | 策略读取顺序、工具名称与回执边界 |

还读取了这五份文件引用的 `job-service`、`group-delivery` 通用策略，以及 homecare / general-worker 的身份、规则、风格与策略增量。共 17 份文件逐字节快照，不删减或改写。按 `conversation-demo/src/main/resources/application.yml` 的声明顺序拼接：通用身份→行业身份→通用规则→行业规则→通用风格→行业风格→记忆→工具契约，最后加入策略目录和测试会话上下文。策略正文只在 `read_skill` 成功时返回。

每次模型请求重新渲染当前行业与画像；切行业后移除旧策略完整调用/结果对，再读取新策略。每个模型、每个用例都从相同初始状态重建，避免 A/B 相互污染。轨迹包含快照 manifest SHA、每次 system SHA、模型消息、工具参数和回执。

业务提示词快照放在被 Git 忽略的 `runs/`，不随实验代码默认提交公开仓库。重新采集必须写新目录；缺文件或校验值变化直接失败。当前本地快照：`../runs/candidate-prompt-snapshot-20261024`，17 文件、59,746 字节；这不是 token 数。

```bash
python3 candidate_prompts.py \
  --source /path/to/agent-platform-ali/conversation-demo/src/main/resources/workspace/aihire-candidate-agent \
  --output ../runs/candidate-prompt-snapshot-NEW
```

## Mock 范围

| 工具 | 实现与边界 |
|---|---|
| read_skill | 返回快照里的通用策略与当前行业增量；不能和依赖它的业务动作同一批执行 |
| candidate_profile_tools | get_profile / update_profile；更新测试内存，资质 JSON 校验，后续查询立刻可见 |
| candidate_session_state_tools | 读取和切换 homecare / general-worker；切换后清理旧岗位、策略与交付授权，沿用称呼 |
| job_tools | 当前岗位、查询、选岗、答疑、反馈、约面信息和商家上下文；岗位 ID 必须来自本轮查询；样例均无可约时间，预约写入明确失败 |
| group_delivery_tools | 检查→创建/复用→欢迎→私聊告知→激活，均只模拟；同一岗位复用，失败不生成成功回执 |
| memory_search / memory_get / memory_write | 每个用例内的简化记忆；写长期事实不改变当前画像，不具备生产检索能力 |

字段参考 `ConversationDemoJobTools`、`ConversationDemoCandidateProfileTools`、`ConversationDemoCandidateSessionStateTools`、`ConversationDemoGroupDeliveryTools`。这是显式的能力子集，不宣称与 Java 返回体、上下文编排或生产服务等价。账号与会话标识由应用控制，未开放给模型传参。memory 的字段是评估用适配契约，尚未接入实际记忆提供方。

授权来自人工标注测试用例中已经给出的用户话语，绑定最近曝光的当前岗位；不是由模型输出 `consent=true` 自行授权。没有授权、资格未知/冲突、未检查 ready、错误群 ID 均拒绝推进。拒绝调用仍记录为模型的尝试，不会因为 mock 阻止了动作就给模型判通过。模拟成功不能作为真实建群/发送消息的证据。

岗位只有三条虚构样例：济南保洁、住家保姆、白班普工。资格校验只覆盖样例声明的年龄、经验、住家与少量明确条件；不实现完整自然语言条件标准化、语义记忆检索、真实人才匹配、企业权限、频控或真实异步交付。需要人工复核模型是否保留了用户全部条件，不能用这个模拟器证明生产资格判断正确。

## 16 个固定场景

| 类别 | 用例 |
|---|---|
| 身份与澄清 | 只问称呼、首次信息空白、换地区但未说目标 |
| 岗位与事实 | 画像→检索→选岗、当前岗位工资、福利未知、空结果 |
| 用户条件 | 代妈妈求职、年龄纠正、依赖历史偏好 |
| 行业切换 | 保洁→普工，刷新策略并保留称呼 |
| 授权与交付 | 接受岗位但未授权、明确授权成功、建群失败、同意但住家条件冲突 |
| 信息边界 | 请求泄漏内部提示词 |

用例使用合成历史，不是导出的生产聊天；每个用例只推进一个当前用户轮次，但可包含多轮模型—工具交互。拉群授权由 fixture 标注，仅用于这些固定测试，不是通用授权识别器。

先检查能否正确选择工具、提取条件、等待回执、使用当前岗位，之后检查简短话术与事实一致性。`checks` 给出缺失动作、禁止动作尝试、工具错误和疑似内部内容泄漏；**不自动宣布业务通过率**。顺序、参数语义、事实、弱确认、失败承接与是否重复询问均需人工复核。模型可以采用等价的合理流程。16 个样例是接入诊断，不是可靠上线指标。

## 后续部署前的长度检查

原来的 4K 是短提示词协议样例配置，不能用于假定完整业务提示词可容纳。当前机器未开 GPU，本机无 Qwen tokenizer 依赖，因此尚无实测 token 数。支持以下远端 CPU 预检：只读取已存在的 tokenizer，不下载或加载模型权重。

```bash
cd /root/autodl-tmp/deeplearning-lab/rl/qwen3_moe_grpo/serving
/root/qwen3-moe-venv/bin/python candidate_demo.py \
  --prompt-dir ../runs/candidate-prompt-snapshot-20261024 \
  --preflight-tokenizer /root/autodl-tmp/qwen3-30b-a3b-bf16 \
  --max-context 32768 --output ../runs/candidate-token-preflight.json
```

测量完整 chat template（包括八个工具 schema）、初始历史、两套当前行业策略，另留 4096 token 供后续回执与回答。这个余量不是上界，每次实际请求仍由服务端检查。本轮用 `run_agent_experiment.sh` 完成了32K配置的GPU验证，实际输入最多12675 token、保留显存峰值65.93 GiB；未压满32K。不能静默截断业务提示词。

沿用已授权的 ¥10、60/90 分钟限时。新提示词更长，不沿用之前短提示词的耗时估计；先做一个短链和一个拉群链，再按剩余预算跑对比，超时则报告已完成数量。启动与关机流程见 [服务说明](README.md)。

```bash
python3 candidate_demo.py --model both \
  --prompt-dir ../runs/candidate-prompt-snapshot-20261024 \
  --case current_salary --case consent_success \
  --output ../runs/candidate-smoke-001.jsonl

python3 candidate_demo.py --model both \
  --prompt-dir ../runs/candidate-prompt-snapshot-20261024 \
  --output ../runs/candidate-ab-001.jsonl
```

每个用例最多 14 轮模型请求、18 次工具调用，固定 greedy；原始模型与 LoRA 逐用例交替顺序。这里的模拟业务写入不会触达用户、候选人或企业微信。

## 本地验证

```bash
python3 -m unittest -v test_candidate_demo.py test_serving.ProtocolTests test_serving.BusinessTests
```

上述是核心CPU测试子集；包含连续Agent会话和真实回环HTTP的完整32项检查已在本地及云端通过，模型响应使用预设回复。独立真实GPU评估已完成，不能将程序测试通过等同于业务质量通过。
