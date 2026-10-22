# 2026-10-09 Books 实验启动记录

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

本机源码：`<本地仓库>/recommendation/generative_books`。

远端源码与数据：`/root/autodl-tmp/deeplearning-lab/recommendation/generative_books`。

已确认机器 RTX 4080 SUPER 32,760 MiB；新任务前无训练进程，数据盘约 67 GiB 空闲。未清理任何旧 ESMM/MMoE 数据、模型或结果。

## 后台任务

- 下载 screen：`books-ranges-20261009`；下载器启动时 PID 5482。
- 总流程 screen：`books-pipeline-20261009`；2026-10-09 10:26:23 +08:00 启动，shell PID 6038。
- 总日志：`runs/pipeline-20261009.log`。
- 阶段：`runs/pipeline_stage.txt`。
- 下载进度：`data/download_status.json`，下载日志 `runs/download-ranges-20261009.log`。
- 各次训练目录：`runs/latest_run.txt` 给出最近的目录。前缀相同，须用各目录 config.json 的 smoke 字段区分真实检查与正式训练。

写入本记录时，总流程处于 `waiting_for_download`。评论文件已核对完成字节段 1,759,051,776 / 3,223,678,899，约 54.57%；元数据文件随后下载。**此文件不是实时状态，也不表示正式训练已经开始**。

后续自动执行：下载校验 → 5 项正确性检查 → 全量数据准备/商品编码 → Books 真实 GPU 短流程 → TIGER、Decoder-only 各最多 5 epochs → 全测试用户评估。任何失败终止总流程并记录 FAILED。

## 启动前验证

- 5 项检查全部通过，远端日志 `runs/preflight-tests.log`。
- 实际 Beauty 小数据短流程：`runs/books-generative-20261009-102132-5564`，已出现两项 CHECKPOINT_RELOAD_OK 及 SMOKE_COMPLETE；没有测试集评估。
- 400,000 个合成商品的容量检查通过：TIGER 4,978,176 参数，峰值分配显存 3,106 MiB；Decoder-only 5,051,904 参数，峰值分配显存 5,254 MiB。是容量检查，不是 Books 准确率或生产延迟。
- 本机与远端的核心 .py/.sh 文件 SHA-256 已核对一致；正式运行会再保存代码快照与哈希。

这轮保留两种模型相同商品编码和样本预算，只比较生成结构；并非官方 OneRec 或完整 TIGER 复现。完整口径和运行参数见 [README.md](README.md)。

## 2026-10-09 11:16 修复后正式启动

原流程于 11:02:43 在商品编码检查处失败：某组三层语义编码对应 569 件商品，超过原第四个消歧 token 的 512 个槽。正式训练尚未开始。已将第四个 token 容量扩至 1024，前三层码本仍各 512，并修正模型词表偏移和混合进制前缀索引。下载结果、文本向量继续复用；旧代码保存在远端 `runs/pre-collision-fix-20261009/`，原失败日志保留。

修复后 6 项检查全部通过，Books 真实短流程的两种模型权重恢复检查及 SMOKE_COMPLETE 通过。重新生成的 367,982 件商品编码全部唯一，最大碰撞组 559，token 容量为 `[512, 512, 512, 1024]`。

- 完整数据：8,898,041 条交互，603,668 个用户，367,982 件商品。
- 每轮训练：7,087,037 个历史前缀样本；固定 20,000 用户验证，每位用户对全量商品库推荐。测试阶段使用全部 603,668 个用户。
- 修复后 screen：`books-pipeline-repaired-20261009`；启动时 shell PID 7896。
- 正式运行：`runs/books-generative-20261009-111624-8085`，`config.json` 中 `smoke=false`。
- 正式日志：`runs/books-generative-20261009-111624-8085.log`。
- 阶段文件：`runs/pipeline_stage.txt`，11:16:24 进入 `formal_training`。
- TIGER 先训练，Decoder-only 随后训练；各最多 5 轮，batch=512，学习率峰值 0.001，warmup=1000 updates，beam=20。

启动后现场确认 TIGER 第 1/5 轮已到第 600/13,842 batch，进程 PID 8091，约 6,213 样本/秒，GPU 利用率 73%，显存 3,930 MiB。此为当时快照，不是实时状态，也没有首轮模型验证结果。热门商品验证基线 Recall@10=0.67%，NDCG@10=0.0035071。模型效果须等待验证输出，不能用训练 loss 代替。

## 2026-10-09 15:33 新启动的 20 轮实验

原 5 轮实验已于 15:07:45 完成训练与 603,668 用户测试，有 FINAL、TRAINING_COMPLETE、summary.json 和 complete 状态。TIGER 测试 Recall@10=0.021197081839686716、NDCG@10=0.012253178736862833；Decoder-only 分别为 0.023587468608572925、0.013731241029608205。两者最佳验证均在第 5 轮。原结果和权重保留。

用户同意增加训练轮数后，启动独立 20 轮预算试验。两种模型从相同 seed=42 重新初始化，非旧权重续训；完整数据、固定语义编码、模型架构和 batch 不变。学习率峰值 0.001，warmup 1,000 updates，余弦衰减至 0.0001 的周期延长到 20 轮，因此前 5 轮的学习率轨迹也不同。每轮在固定 20,000 验证用户上按 NDCG@10 选模，连续 3 轮不提升早停，两种模型选择完毕再全量测试。

- 启动脚本：`start_20epochs.sh`，复用 pipeline.lock，语法检查通过；训练器、模型和原启动脚本本地/远端 SHA-256 一致。
- 后台 screen：`books-20epochs-20261009`，启动时 screen PID 14522，训练 PID 14539。
- 正式运行目录：`runs/books-generative-20261009-153342-14533`，已确认 `epochs=20`、`smoke=false`、`resume=false`。
- 正式日志：`runs/books-generative-20261009-153342-14533.log`。
- 外层日志：`runs/pipeline-20epochs-20261009.log`。
- 当前阶段文件：`runs/pipeline_stage.txt`；当前目录：`runs/latest_run.txt`。
- 15:34:56 快照：TIGER 第 1/20 轮第 800/13,842 批，约 6,805 样本/秒，batch loss=4.2035，GPU 显存 3,930 MiB；Decoder-only 尚未开始。
- 若均跑满，按原实验吞吐估计总耗时约 15 小时；可能因早停缩短。快照不代表实时状态，后续应读取新目录日志和进程。

模型数据签名中的 semantic_ids SHA-256 仍为 `46c9399b6e58555801bb9b00f11ee3d4bedff7471b42ce0cbb8b51479d3513f2`。启动前空闲显存充足，数据盘 61 GiB 可用，无旧训练进程。未删除旧 ESMM/MMoE 数据。

## 2026-10-10 检查：Decoder-only 失败，TIGER 完成并独立测试

20 轮整组流程于 2026-10-10 00:35:20 退出，`status.json` 为 failed，错误 `RuntimeError('Nonfinite gradient')`。TIGER 已于前一晚完成 20 轮，按验证 NDCG@10 选中第 18 轮：验证 Recall@10=0.0407、NDCG@10=0.02623175536797882。

Decoder-only 完成 6 轮，在第 7 轮中断；最后一条 STEP 为 batch 6400/13842。第 5 轮是最佳验证权重（Recall@10=0.0264、NDCG@10=0.015640602569000107）。第 6 轮 batch 6800 附近开始观察到大幅损失升高，整轮验证 Recall@10 跌至 0.00005。第 6 轮 last.pt 的权重与优化器张量仍有限，但模型已明显退化；不可因文件可加载就视为健康恢复点。未启动新的训练，失败根因尚未完全定位，不能把高学习率假设说成已验证结论。

为回答结果查询，单独评测已完成且已选定的 TIGER。脚本 `results/evaluate_completed.py` 使用运行目录保存的 source，校验源文件/数据签名与最佳 checkpoint，先完整复现原 20,000 用户验证指标，然后评测全部 603,668 测试用户。保持原 failed 状态，不写整组完成 summary。评测日志出现 INDEPENDENT_EVALUATION_COMPLETE；没有重训或选择其他权重。

- 最佳 TIGER checkpoint SHA-256：`9c3b09b6ac0766297a5f945c4a166afdf996c8b280ce03449f4a99ac1861f894`。
- TIGER 测试 Recall@10=0.033337861208478835，NDCG@10=0.020612178234259957，Recall@20=0.042125804250018224。
- 测试用户哈希与 5 轮基线一致：`7c853e028989c14d979f489efad84eac5bc2cc746a6037a16e7f0a9871618ae3`。
- 原运行目录下新增 `tiger_test.json`、`tiger_test_predictions.npz`、`tiger_independent_evaluation.json`、`tiger-independent-evaluation.log`。
- 本机指标与验证历史保存在 `results/books-generative-20261009-153342-14533/`。Decoder-only 没有本轮最终测试成绩，不能汇报为完成 20 轮。

## 2026-10-10 09:25 Decoder-only 低学习率恢复已启动

用户明确授权修改参数后重新跑。新实验只训练 Decoder-only，从失败运行第 5 轮验证最佳权重恢复，重新初始化 AdamW（原 best.pt 不含优化器状态），最多再跑 15 轮，累计第 6～20 轮。原训练器、已完成 TIGER、原失败现场与旧结果均保留。

- 新脚本：`decoder_recovery/train_recovery.py`、`decoder_recovery/start_train.sh`，测试：`decoder_recovery/test_recovery.py`。
- 峰值学习率 0.0001，前 1,000 updates 从 0.00001 预热到峰值，之后余弦下降至 0.00001。
- batch=512，BF16，clip=1，weight_decay=0.01，seed=42；全部 7,087,037 训练样本，固定 20,000 验证用户、beam=20、NDCG@10 选模、patience=3。
- 基线第 5 轮也参与最佳权重选择；训练结束对最佳权重进行全量 603,668 用户测试。
- 4 项针对性检查通过；真实 Books 短流程在第一个 epoch 后停止，再恢复完成第二个，最佳 checkpoint 验证复现通过，未使用测试集。短流程目录 `runs/decoder-recovery-smoke-20261010-a`。
- 正式目录：`runs/books-decoder-recovery-lr1e-4-20261010-092530-41355`。
- 正式日志：`runs/books-decoder-recovery-lr1e-4-20261010-092530-41355.log`。
- 外层日志：`runs/pipeline-decoder-recovery-20261010.log`。
- screen：`books-decoder-recovery-20261010`；启动时训练 PID 41365。
- 09:25:53 已确认 `smoke=false`，完整复现父 checkpoint 验证 Recall@10=0.0264、NDCG@10=0.015640602569000107，进入 `recovery_epoch=1`、`cumulative_epoch=6`。首批 loss=2.844486、梯度范数=0.500903，GPU 利用率 96%、显存 6,824 MiB。
- 来源 best.pt SHA-256：`1504792ba4acf87a2dcc58846466d78be6ef11e8b677840f72decb3ae4bddedf`。

每 200 批记录裁剪前梯度范数和最大值，检查参数/优化器有限性；异常保留逐层报告与批次状态，不跳过错误批次。新实验自身支持完成 epoch 后恢复优化器、scheduler 与 RNG。此处为启动快照，当前状态需读取新目录 status.json 和进程；原 20 轮运行仍为 failed，不要误读其状态为新任务状态。

## 2026-10-10 15:18 Decoder-only 恢复实验完成

15:32 现场核对：阶段 `complete_decoder_recovery`，新目录 status.json 为 complete，日志有 EARLY_STOP 19、FINAL 与 TRAINING_COMPLETE；无训练进程，GPU 利用率 0%、显存 1 MiB。实际于 15:18:09 完成。这次未再次出现非有限梯度异常。

累计训练到第 19 轮（父模型 5 轮 + 本次 14 轮），第 17、18、19 轮验证 NDCG@10 连续三轮没有超过第 16 轮，因此正常早停。最佳权重来自累计第 16 轮：验证 Recall@10=0.03365、NDCG@10=0.020417585455320256。

所选最佳权重已完成 603,668 用户全量测试：Recall@10=0.02694030493582565，NDCG@10=0.016027854547466424，Recall@20=0.03501924899116733，Top-10 命中 16,263 次。测试用户 SHA 与旧 5 轮及 TIGER 一致。相对旧 5 轮 Decoder-only，Recall@10 提升 14.21%、NDCG@10 提升 16.73%。本轮恢复训练和评测耗时约 5 小时 53 分钟。

当前已测最优仍为 TIGER 20 轮预算所选第 18 轮权重，测试 Recall@10=0.033337861208478835、NDCG@10=0.020612178234259957。两种架构训练轨迹不同且为单随机种子，不能将差距全部归因于结构。新 summary.json 与 history.json 已同步到本机 `results/books-decoder-recovery-lr1e-4-20261010-092530-41355/`。

## 2026-10-10 15:44 TIGER 宽度扩大实验已启动

用户授权增大 TIGER 网络后重新训练。新目录 `tiger_scaling/` 从原已验证的模型/训练器派生，旧代码和权重不改；只训练新 TIGER，不重跑 Decoder-only。

结构：d_model 128→256，d_ff 1024→2048；Encoder 4 层、Decoder 4 层、6 个注意力头、每头 d_kv=64 均不变。参数 5,043,712→14,281,344（2.8315 倍）。从 seed=42 随机参数开始，非原权重续训。

训练与原 TIGER 20 轮基线一致：全量 7,087,037 样本/轮，batch=512，BF16，AdamW 峰值 lr=0.001，warmup 1,000 updates，20 轮余弦衰减到 0.0001；weight_decay=0.01、clip=1、dropout=0.1，固定同一批 20,000 验证用户、beam=20、patience=3。保留梯度日志，非有限梯度在裁剪前拦截，并保存故障批次。全量测试在选模完成后进行。

- 3 项结构/生成/梯度与 checkpoint 检查通过。
- 实际全商品库容量检查：batch512，峰值 allocated=4,979 MiB，reserved=5,280 MiB，短时吞吐约 4,422 样本/秒。新参数量 14,281,344。
- 真实 Books 短流程 `runs/tiger-w256-smoke-20261010-a` 出现 CHECKPOINT_RELOAD_OK 和 SMOKE_COMPLETE，未用测试集。新旧 data_signature 完全一致，正式 validation_users.npy 字节完全相同。
- 正式目录：`runs/books-tiger-w256-20epochs-20261010-154410-50773`。
- 正式日志：`runs/books-tiger-w256-20epochs-20261010-154410-50773.log`。
- 外层日志：`runs/pipeline-tiger-w256-20261010.log`。
- screen：`books-tiger-w256-20261010`；启动时训练 PID 50782。
- 启动脚本：`bash tiger_scaling/start_train.sh`，与其他 Books 任务共用 pipeline.lock。
- 15:46 快照：第 1/20 轮，batch800/13842，loss=3.9895、裁剪前梯度范数=0.4148，约 4,319 样本/秒；GPU 88%，显存 5,652 MiB。实际配置 smoke=false。

按当时吞吐若跑满 20 轮约 9～10 小时，可能早停。此为启动快照，进度应查最新目录的 status.json 与日志；暂没有新模型验证或测试成绩。单随机种子、扩大主宽度与 FFN 的实验，不同时增加深度。具体实现和恢复命令见 `tiger_scaling/README.md`。

## 2026-10-10 22:57 用户要求暂停宽版 TIGER

用户认为本轮效果不好，明确要求先暂停。已核对 PID 50782 的精确运行目录和命令后发送 SIGTERM，进程退出且 pipeline.lock 释放；未删除任何模型、日志或数据。原任务 status.json 与 pause.json 已记录 paused，公共阶段文件为 paused_tiger_w256_by_user，日志追加 TRAINING_PAUSED_BY_USER。

暂停前最后记录为第 16 轮 batch 7600/13842；最后完整 checkpoint 是第 15 轮，best_epoch 也为 15。last.pt 已确认包含模型、优化器、scheduler、CPU/CUDA RNG。第 16 轮未完成的进度不在 checkpoint 中，未来若用户要求恢复，应从第 16 轮开头重跑。不要自行重新启动或标记任务已完成；本轮没有最终测试结果。
