# 历史环境与模型来源

2026-10-22 从 `deeplearning-dl-1` 取回，不包含依赖包、模型或训练数据。

- `generative-20261009.txt`：Beauty 的 `beauty-tiger-20261009-091820-3491` 与 Books 的 `books-generative-20261009-111624-8085` 两份 `environment.txt` 字节相同，SHA-256 为 `74d5da26d2f6672f26ae14d6c677fa96436eec31aea13f167a908c0ae3383717`，去重保存一份。包含 PyTorch 2.8.0+cu128、Transformers 4.57.6、sentence-transformers 5.7.0、scikit-learn 1.9.1。`file:///croot/...` 是原镜像的构建路径，此文件是运行证据，不是跨机器可直接安装的 requirements。
- `beauty-content-download-manifest.json`：Beauty 数据与 Sentence-T5-base 内容编码器的历史下载 URL、字节数和 SHA-256。Books 复用了这一编码器。URL 使用 `master`，可能漂移；应以记录的哈希核对文件身份。此次只核对清单文件本身，未重新读取全部大文件验证哈希。
- Ali-CCP 下载环境的固定包版本位于 `recommendation/esmm_aliccp_full/download_requirements.txt`，同样来自远端保存记录。

完整的远端路径与本地文件映射见 [迁移清单](../migrations/2026-10-22/source_inventory.json)。上述记录不等同于完整、可移植的环境锁定文件。
