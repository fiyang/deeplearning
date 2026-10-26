# 招聘 Agent 部署与连续对话

本次选择仓库里的轻量 Python Agent，接入完整 AIHire C 端提示词。它在模型服务外管理会话、当前画像、岗位、业务策略、工具执行和回执；不是只有一个 system prompt 的裸聊天请求。未安装 OpenClaw，也未启动 Java Harness。八个业务工具仍使用隔离的 mock。

## 当前验证状态

- Agent、模型服务接口、终端客户端和限时实验启动器已实现。
- 32 项本地测试通过，包含真正的回环 HTTP 请求、连续会话、鉴权、画像保留与工具回执；模型返回值为预设测试回复。
- 2026-10-24 11:12，A800 上模型服务和 Agent 均已启动；32 项云端程序检查通过。完整提示词、两个技能和 4096 token 预留的最大合计为 18431 token，选择 32768 上下文。
- 本轮运行目录：`runs/qwen3-agent-20261024-111119-1725/`。6 个连续用户轮次和32次场景评估已完成，业务流程存在明显缺失；两组自动检查均仅4/16未标记问题，不等于业务通过率。见[完整实测](RESULTS_20261024.md)。实例已关机，当前服务离线。

## 结构

```text
agent_chat.py（连续终端聊天）
  → agent_server.py 127.0.0.1:8001（会话、策略、工具循环）
  → serve.py 127.0.0.1:8000（Qwen3 BF16，原始 / LoRA）
  ← 模型生成回答或 tool_calls
  → mock 执行 + 工具回执 → 模型继续 → 最终话术
```

Agent 每个会话固定使用原始模型或第二轮 LoRA。每轮重新注入当前行业和画像，历史保留完整工具调用/回执。切行业移除旧策略；失败轮次保留诊断轨迹并终止该会话，避免重放不完整的动作链。最多 12 个用户轮次，每轮最多 14 次模型请求、18 次工具调用；实际上下文超限明确报错，不静默删除业务提示词。

会话内工具不会执行 shell、不连接真实招聘服务、不发企业微信。拉群授权识别是本地实验的保守匹配，只识别少量明确语句和唯一拉群提问后的确认；它不是可直接上线的授权识别器。岗位曝光识别也是 mock 检查，真实系统应改接已有确定性业务状态和回执。

## 云端限时实验

已授权本次算力不超过 ¥10，A800 已核实 ¥5.59/小时。开机时间必须取实际值，平台定时关机必须独立设置且不晚于开机后 90 分钟。程序自身在开机后 60 分钟内结束，完成或失败即关机。时间计算包括部署、模型加载和评估；不自动租机、不自动延长服务。

```bash
cd /root/autodl-tmp/deeplearning-lab/rl/qwen3_moe_grpo/serving
export QWEN3_SERVE_APPROVED_BUDGET_CNY=10
export QWEN3_PRICE_CNY_PER_HOUR=5.59
export QWEN3_INSTANCE_STARTED_AT=ACTUAL_BOOT_UNIX_SECONDS
export QWEN3_PLATFORM_DEADLINE=VERIFIED_SHUTDOWN_UNIX_SECONDS
bash run_agent_experiment.sh
```

启动器先用已有 tokenizer 在 CPU 上测完整提示词与工具 schema，再选择 16K 或 32K（按实测长度加余量），启动模型和 Agent。先对原始模型、LoRA 分别进行“称呼→找保洁→追问税前/到手”的连续三轮对话，然后进行 16 场景 A/B。任何完成/未完成数量都写入 summary；不会把中途结果算作全量成功。本轮实测请求约9.8K～12.7K token，最高保留显存65.93 GiB；未验证压满32K或并发请求。

日志：

```bash
tail -f /root/autodl-tmp/deeplearning-lab/rl/qwen3_moe_grpo/runs/latest_agent.log
```

每次运行建立独立 `runs/qwen3-agent-时间-PID/`，包含 `preflight.json`、`model.log`、`agent.log`、`experiment.log`、`summary.json`、A/B 与逐会话轨迹。会话 key 存在该目录的 `api_key` 文件，权限 600，不打印、不提交仓库。

## 服务运行期间手动聊天

在另一远端终端执行（用本次实际运行目录替换 RUN_DIRECTORY）：

```bash
cd /root/autodl-tmp/deeplearning-lab/rl/qwen3_moe_grpo/serving
export QWEN3_API_KEY="$(cat RUN_DIRECTORY/api_key)"
/root/qwen3-moe-venv/bin/python agent_chat.py --model qwen3-grpo
```

可依次输入“我35岁，在济南找保洁”“这个工资是到手的吗”“先了解，不拉群”。调用原始模型时改为 `--model qwen3-base`。自动评估同时在使用单卡时可能返回忙，请勿并行压测。

Agent API 为 `POST /sessions` 和 `POST /sessions/{session_id}/messages`，必须使用同一个 Bearer key。消息请求只接受 `text`，不接受客户端伪造 session metadata 或授权标记。服务绑定本机回环；若使用本地电脑聊天，需恢复 SSH 登录后建立端口转发。当前实验的预算不包含长期常驻服务。

业务提示词、工具范围与场景见 [场景说明](CANDIDATE_SCENARIO.md)。
