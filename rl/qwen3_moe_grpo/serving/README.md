# Qwen3 招聘工具调用实验

连续会话的 Agent、终端聊天与限时部署入口见 [Agent 部署说明](AGENT_DEPLOYMENT.md)。

状态：2026-10-24 已在 A800 完成真实模型部署与测试：6 个连续用户轮次、16场景×原始/LoRA两组。接口和mock工具调用可用，但业务流程跟随不足，详见[实测记录](RESULTS_20261024.md)。实例已关机。主入口为 [C 端求职顾问实验](CANDIDATE_SCENARIO.md)，下文三工具 demo 仅保留作协议接线样例。

复用已有 Qwen3-30B-A3B BF16 底座和第二轮最终 adapter，沿用已完成数学评估的 Unsloth/PEFT 加载及逐张量校验。不会合并/覆盖底座，也不会新增一份约 61 GB 的合并权重。注意力和 MoE 专家 adapter 均保留。单卡一次处理一个请求，通过请求中的 model 字段串行比较：

- `qwen3-base`：同一底座，临时禁用 adapter。
- `qwen3-grpo`：启用第二轮最终 adapter。

模型只生成回答和函数调用建议。业务程序校验函数名、参数、账号范围后执行工具，把结果作为 tool 消息回传，模型再组织回答。工具返回的数据不授予新的权限。本地 mock 数据全为虚构，无真实候选人信息、无消息发送、无数据库写入。匹配只演示城市和技能交集查询，不是招聘决策模型。

```text
用户请求
  → 业务程序加入工具 schema 和固定账号范围
  → 推理服务（底座 / LoRA）生成 tool_calls
  → 程序校验并执行 search_jobs / match_candidates / get_recruiting_progress
  → 将工具结果回传模型
  → 用户可读回答 + 完整调用轨迹
```

## 首版服务范围

标准库 HTTP 接口：`GET /health`、`GET /v1/models`、`POST /v1/chat/completions`。采用常见 chat completions 字段及 Hermes 工具格式，**只实现一个明确的子集**，不是完整 API/生产网关。支持普通文本、tools、auto/none、非流式请求；不支持 streaming、强制指定工具、图片、n>1、任意扩展字段。未知字段和超长上下文直接报错，忙时返回 429。

默认 BF16、非 thinking、4096 总上下文、最多 1024 输出 tokens、并发 1。4096 仅是短提示词样例默认值。完整 C 端提示词需先做 tokenizer 预检，并显式选择更长上下文（支持至 32768）；长上下文显存、延迟和 adapter 开关仍需 GPU 短跑，不静默截断业务提示词。服务只监听 `127.0.0.1`，使用 `QWEN3_API_KEY`，不直接暴露公网。server 日志仅记 token 数和耗时；demo 轨迹保存在被 Git 忽略的 `runs/`。

首版复用 `/root/qwen3-moe-venv`，不安装 vLLM，不修改训练环境，不下载新权重。最新 vLLM 文档已说明 MoE LoRA 的 2D/3D 布局兼容选项；将来迁移时应检查实际 adapter 张量键与布局、固定 vLLM 版本，做底座和 adapter 输出回归后再进行并发性能测试，不能直接假设导出的 expert adapter 已兼容。

## 启动流程（批准预算后在远端执行）

1. 复用 `deeplearning-dl-3` 的 A800 80GB，核对实时价格与磁盘。设置开机后不超过 90 分钟的平台关机兜底，记录实际开机和关机时间戳。
2. 将本目录连同现有 `common.py`、`round2.py`、`rl/qwen7b_grpo/rewards.py` 放到远端仓库对应路径。代码传输量很小；不经本机传输模型/数据。不要运行训练环境 bootstrap。
3. 生成会话 key，填写本次实际授权预算与时间。下列大写占位值需替换，不能直接原样运行：

```bash
cd /root/autodl-tmp/deeplearning-lab/rl/qwen3_moe_grpo/serving
export QWEN3_API_KEY="$(/root/qwen3-moe-venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export QWEN3_SERVE_APPROVED_BUDGET_CNY=APPROVED_BUDGET
export QWEN3_PRICE_CNY_PER_HOUR=VERIFIED_PRICE
export QWEN3_INSTANCE_STARTED_AT=ACTUAL_BOOT_UNIX_SECONDS
export QWEN3_PLATFORM_DEADLINE=CONFIRMED_SHUTDOWN_UNIX_SECONDS
bash run_service.sh
```

启动器把加载模型时间计入开机后的 60 分钟服务时限，并进一步按预算与平台截止时间收紧；退出或异常后关机。平台兜底必须独立设置。它不会自动租机器或付费开机。现有扩容盘费用按原规则继续。

4. 在第二个远端终端设置同一 `QWEN3_API_KEY`（不要把 key 放进仓库）。本轮执行 [candidate_demo.py 的预检、短跑与对比](CANDIDATE_SCENARIO.md)。以下为旧协议样例的备用命令：

```bash
cd /root/autodl-tmp/deeplearning-lab/rl/qwen3_moe_grpo/serving
python3 recruiting_demo.py --model both --output ../runs/recruiting-ab-001.jsonl
```

固定 10 个场景 × 两种模型；每场景最多 5 次模型请求、6 次工具调用。包含查职位、匹配、进度、多工具串联、空结果、无权限、缺信息澄清、写操作请求、闲聊和禁止凭空回答。

```bash
python3 recruiting_demo.py --model qwen3-grpo \
  --prompt '先查北京的Java职位，再查询最匹配的2名候选人和招聘进度。' \
  --output ../runs/recruiting-interactive-001.jsonl
```

本机体验可建立隧道：`ssh -N -L 8000:127.0.0.1:8000 deeplearning-dl-3`，然后在本机运行同一个 demo。2026-10-24 读取筛选结果时 SSH 密钥认证未通过，使用的是 JupyterLab；本机隧道需要先恢复 SSH 登录。未恢复时可在 JupyterLab 的两个终端启动服务和客户端。

## 如何评价效果

先比较相同业务提示、相同工具和相同 mock 数据下的底座 / LoRA；评估客户端固定 greedy。原来的数学 `<answer>` system prompt 不用于业务对话。数学准确率 93.33% 不能解释为工具调用正确率，数学 GRPO 也可能使业务表现退化。

记录和人工复核：

| 指标 | 口径 |
|---|---|
| 工具选择与参数 | 是否调用该调用的工具、ID/城市/数量是否正确、是否先查到职位再用其 ID |
| 端到端业务成功 | 调用结果及最终文字是否满足请求；空结果、无权限、缺参数是否处理正确 |
| 事实忠实度 | 人数、候选人、状态是否与工具结果一致，有无虚构 |
| 运行约束 | 未知工具、非法参数、越权请求是否由程序阻止，有无循环、截断、超时 |
| 性能 | 服务端首 token 耗时、总生成时间、输出 tokens/s、显存、客户端完整业务耗时 |

`checks` 只给严格的参数匹配和答案关键词提示，不把它当完整业务成功率。等价的关键词（如“数据”与“数据工程师”）可能被严格匹配判为失败；数字出现也不代表事实对应正确。10 个场景适合发现接线和基础行为问题，不足以给出可靠上线通过率。首 token 时间来自生成回调，当前 HTTP 不流式传输，用户实际收到答案的时间看完整耗时。tokens/s 包含服务端前处理/生成开销，不是纯解码吞吐；首个请求可能有预热成本。

模拟链路通过后再将这三个工具的实现替换为现有业务只读接口，保留 schema、服务端账号范围与调用轨迹。若工具选择或参数抽取差，再收集经过审核的业务轨迹做工具调用 SFT/GRPO；业务奖励应围绕可验证的参数、工具结果和任务完成，不能继续沿用纯数学答案奖励。

## 本地检查

```bash
cd rl/qwen3_moe_grpo/serving
python3 -m unittest -v test_serving.py
```

仅协议、mock 工具、HTTP 接线和预算限制相关的 CPU 测试；没有加载 Qwen3，不能证明模型工具调用质量或 A800 服务可用。

参考：

- [Qwen 官方函数调用文档](https://qwen.readthedocs.io/en/latest/framework/function_call.html)：Qwen3 内置 Hermes 工具模板、函数调用循环。
- [vLLM LoRA 文档](https://docs.vllm.ai/en/latest/features/lora/)：MoE adapter 的 2D/3D 布局说明。
- [Qwen3-30B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B)：原始模型及推理说明。
