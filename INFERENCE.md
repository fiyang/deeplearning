# ViT 推理服务与实测结果

> 历史实验记录：下文结果、路径和运行状态来自此前实验；2026-10-20 归档，尚未与远端最终文件核对。数据、图表与模型产物不随 Git 分发。

2026-10-06 已在 AutoDL 启动。2026-10-07 抽检时发现进程与本机通道已退出，使用相同最佳权重恢复了服务和 SSH 通道，并完成准确率检查。模型常驻 GPU，上传图片即可获得分类分数，也可取出 192 维特征。

## 部署的是哪一个模型

依据验证集成绩选择 **12 层、192 维、600 轮**实验的最佳权重，保存于第 **543** 轮。验证准确率 **93.94%**，测试准确率 **93.68%**，参数量 **5,362,762**。测试准确率比原 300 轮基线提高 3.12 个百分点，比 384 维 / 300 轮模型提高 2.80 个百分点。

- 框架：PyTorch 2.8.0+cu128，FastAPI 0.142.2，Uvicorn 0.54.0。
- 硬件：RTX 4080 SUPER，驱动 595.71.05。
- 推理精度：FP32，与模型评估精度一致；没有量化。
- 图片预处理：转 RGB、缩放到 32×32、按训练配置归一化。
- 权重：`/root/autodl-tmp/deeplearning-lab/runs/vit-scaling-20261006-200157-1726/width192-600epochs/best.pt`。
- 权重 SHA256：`eb17ba6bd948093715b80ea2abad0b2431ea7527bc79b8206dab5dcb67786466`。

识别范围是 CIFAR-10 的十类：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。93.68% 是 CIFAR-10 测试集上的成绩；自己的照片仍会被归入这十类之一。返回的 `score` 是 softmax 分数，未经置信度校准。

## 从本机调用

当前本机地址为 **http://127.0.0.1:18080**，通过 SSH 转发到服务器的 `127.0.0.1:8008`。服务只监听服务器本地地址，访问由 SSH 登录提供保护。

健康检查：

```bash
curl -sS http://127.0.0.1:18080/health | python3 -m json.tool
```

发送随项目保存的样例：

```bash
curl -sS http://127.0.0.1:18080/predict \
  -H 'Content-Type: image/png' \
  --data-binary @<本地仓库>/examples/cifar10-sample.png \
  | python3 -m json.tool
```

本次实际返回的前三类为 `cat: 0.789713`、`ship: 0.100814`、`airplane: 0.027031`。图片请求体直接传二进制，不是 JSON，也不是 multipart 表单。支持 PNG、JPEG、WebP；最多 8 MiB、1600 万像素。把上述路径换成自己的图片即可。

需要特征向量时，把 URL 改成：

```text
http://127.0.0.1:18080/predict?top_k=3&include_features=true
```

`top_k` 支持 1–10；返回的 `features` 为长度 192 的浮点数列表。默认只返回分类结果和耗时，不传特征。接口文档为 [本机 API 文档](http://127.0.0.1:18080/docs)。

## 性能实测

下列测试顺序执行，GPU 没有训练任务干扰。使用同一张 2493 字节的 CIFAR-10 PNG，每次请求一张图片（batch=1），HTTP 连接复用，测量前预热。服务器每组 500 次正式请求、50 次预热；Mac 每组 100 次正式请求、10 次预热。

| 调用位置 | 并发数 | 请求数 | 平均端到端延迟 | P95 延迟 | 吞吐量 | 失败数 |
|---|---:|---:|---:|---:|---:|---:|
| 服务器内 HTTP | 1 | 500 | 7.44 ms | 8.26 ms | 133.4 张/秒 | 0 |
| 服务器内 HTTP | 4 | 500 | 27.79 ms | 31.90 ms | 143.1 张/秒 | 0 |
| Mac 经 SSH | 1 | 100 | 79.22 ms | 96.67 ms | 12.6 张/秒 | 0 |
| Mac 经 SSH | 4 | 100 | 98.30 ms | 187.31 ms | 37.6 张/秒 | 0 |

P95 表示约 95% 的请求耗时不超过该值。吞吐量按成功请求数除以整组测试时间计算。这是固定图片和有限并发下的短测结果，不是服务最大容量或长期稳定性保证。

服务器内单并发时，前向的 CUDA 事件计时平均 **4.78 ms**，包含预处理、调度与后处理的服务内部计时平均 **5.86 ms**；HTTP 端到端为 **7.44 ms**。CUDA 事件测量的是前向开始到结束的设备时间区间，不是全部请求耗时。服务内部计时不含返回 JSON 的序列化及响应网络传输；本机端到端计时包含 SSH 和网络往返。不同请求节奏下模型内部时间也会变化，不能把两组 HTTP 延迟之差全部当成固定网络延迟。

当前服务采用一个常驻模型、单 worker，推理串行执行，并发请求排队，未启用动态合批。因此并发从 1 增至 4 时，吞吐量仅小幅提高而单次延迟增加。上传更大的图片、返回特征或使用不同网络时，耗时会变化。

服务 GPU 进程的 `nvidia-smi` 占用约 **350 MiB**，整卡读数约 **360 MiB**；PyTorch 缓存保留量约 **44 MiB**。这些指标包含的内容不同，不能混为模型权重大小。

原始逐请求记录位于 `runs/inference-20261006-230556/`：`server-c1.json`、`server-c4.json`、`mac-c1.json`、`mac-c4.json`。全部 **1200 次**正式测量请求成功，另有接口检查和预热请求未计入表格。

## 已验证的行为

`check_inference_service.py` 实际调用服务，验证分类、分数及完整特征向量与原离线推理一致；12 次并发调用结果一致；空请求、非法图片、超大请求及非法 top_k 被正确拒绝；灰度、JPEG、RGBA PNG、WebP 均能处理。检查记录为 `runs/inference-20261006-230556/service_checks.json`。

部署源码快照、校验值、环境版本、GPU 记录及本机调用示例也保存在该目录。原训练环境未修改；API 组件安装在独立的 `.venv-inference` 中。

## 准确率抽检（2026-10-07）

从原 CIFAR-10 测试集中按类别随机抽取，每类 50 张，共 500 张，种子为 20261007。抽样名单在预测前保存，图片以无损 PNG 逐张提交真实 HTTP 接口，没有根据结果重新挑选样本。

| 指标 | 结果 |
|---|---:|
| HTTP 请求成功 | 500 / 500 |
| 第一预测类别正确 | **466 / 500（93.2%）** |
| 前三类包含真实类别 | 494 / 500（98.8%） |
| 同批图片离线准确率 | 93.2% |
| 接口与离线第一预测一致 | **500 / 500（100%）** |
| 类别分数最大绝对差异 | 0.00000197 |
| 原完整 10,000 张测试集成绩 | 93.68% |

这次抽检未发现服务部署导致的分类结果变化。500 张子集与完整测试集的准确率有所不同，不能把 93.2% 与 93.68% 的差别当作精度下降；两条推理路径在同一批图片上的准确率和分类结果完全一致。

| 类别 | 正确数 / 50 | 准确率 |
|---|---:|---:|
| 飞机 | 49 | 98% |
| 汽车 | 48 | 96% |
| 鸟 | 46 | 92% |
| 猫 | 40 | 80% |
| 鹿 | 48 | 96% |
| 狗 | 43 | 86% |
| 青蛙 | 49 | 98% |
| 马 | 47 | 94% |
| 船 | 49 | 98% |
| 卡车 | 47 | 94% |

本次猫和狗相对较弱，每类只有 50 张，类别百分比仅供定位问题。此次使用的是既有测试集的子集，不是新增独立测试集，也不代表任意手机照片上的准确率。

报告、抽样索引、逐张结果、检查脚本快照及图例位于 `runs/inference-accuracy-20261007-0830/`。图例按事先规定的规则展示每类抽中的第一张，以及前六个错误案例。

![接口准确率抽检样例](runs/inference-accuracy-20261007-0830/prediction_examples.png)

复现检查（输出目录需使用新名称）：

```bash
cd /root/autodl-tmp/deeplearning-lab
.venv-inference/bin/python check_inference_accuracy.py \
  --run runs/vit-scaling-20261006-200157-1726/width192-600epochs \
  --per-class 50 --seed 20261007 --output runs/my-inference-accuracy-check
```

## 管理与复现

当前服务后台会话：`vit-inference-8008-20261007-082810-1484`。
日志：`/root/autodl-tmp/deeplearning-lab/runs/vit-inference-8008-20261007-082810-1484.log`。
当前进程 PID 为 1491；重启后的进程号以 `/health` 返回值为准。

查看日志：

```bash
ssh deeplearning-dl-1 'tail -f /root/autodl-tmp/deeplearning-lab/runs/vit-inference-8008-20261007-082810-1484.log'
```

服务器重启后，在服务器重新启动：

```bash
cd /root/autodl-tmp/deeplearning-lab
bash start_inference.sh
```

默认加载已选中的 192 维 / 600 轮最佳权重。也可将其他兼容的最佳权重路径作为第一个参数。端口已被占用时会报错，避免启动重复服务。

SSH 通道断开后，在 Mac 的一个终端里重新连接，并保持终端开启：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18080:127.0.0.1:8008 deeplearning-dl-1
```

当前后台通道控制文件为 `/private/tmp/deeplearning-inference-18080-20261007.sock`。检查或关闭该通道：

```bash
ssh -S /private/tmp/deeplearning-inference-18080-20261007.sock -O check deeplearning-dl-1
ssh -S /private/tmp/deeplearning-inference-18080-20261007.sock -O exit deeplearning-dl-1
```

关闭 SSH 通道只关闭本机入口，服务器推理服务仍在后台。停止本次服务器服务的命令为：

```bash
ssh deeplearning-dl-1 'screen -S vit-inference-8008-20261007-082810-1484 -X quit'
```

重新测服务器单并发性能（输出文件需用新名字，脚本不覆盖已有记录）：

```bash
cd /root/autodl-tmp/deeplearning-lab
.venv-inference/bin/python benchmark_inference.py \
  --image examples/cifar10-sample.png --requests 500 --concurrency 1 --warmup 50 \
  --context server-loopback-idle-GPU --output runs/my-inference-benchmark.json
```

重建独立推理环境：

```bash
/root/miniconda3/bin/python -m venv --system-site-packages .venv-inference
.venv-inference/bin/python -m pip install -r requirements-inference.txt
```

模型在服务启动时加载并预热，相关实现参考 [FastAPI lifespan 文档](https://fastapi.tiangolo.com/advanced/events/)；设备侧计时使用 [PyTorch CUDA Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)。
