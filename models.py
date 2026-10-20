"""同一张 32×32 图片，可以由 CNN、ResNet 或 ViT 提取特征。"""
import torch
from torch import nn
from torchvision.models import resnet18

MODEL_SPECS = {
    "smallcnn": {"label": "Small CNN", "feature_dim": 256, "dropout": 0.3},
    "resnet18": {"label": "ResNet-18 (CIFAR stem)", "feature_dim": 512,
                 "stem": "3x3 convolution, stride 1; no initial max pool"},
    "vit": {"label": "Small ViT", "feature_dim": 192, "patch_size": 4,
            "patch_count": 64, "depth": 6, "heads": 3, "mlp_dim": 768, "dropout": 0.1},
    "vit_tiny12": {"label": "ViT 12-layer (CIFAR)", "feature_dim": 192, "patch_size": 4,
                   "patch_count": 64, "depth": 12, "heads": 3, "mlp_dim": 768, "dropout": 0.0},
    "vit_small12": {"label": "ViT 12-layer 384-wide (CIFAR)", "feature_dim": 384, "patch_size": 4,
                    "patch_count": 64, "depth": 12, "heads": 6, "mlp_dim": 1536, "dropout": 0.0},
}


class CifarResNet18(nn.Module):
    """ResNet 也是 CNN。残差连接帮助较深的网络训练。"""
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None, num_classes=10)
        # ImageNet 原版的第一层缩小尺寸较快，这里适配 32×32 的小图片。
        self.backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.backbone.maxpool = nn.Identity()

    def forward_features(self, images):
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return net.avgpool(x).flatten(1)  # 每张图片 -> 512 个特征数值。

    def forward(self, images):
        return self.backbone.fc(self.forward_features(images))


class SmallViT(nn.Module):
    """把图片切成 64 个 4×4 小块，使用 Transformer 编码图像特征。"""
    def __init__(self, depth=6, dropout=0.1, embed_dim=192, heads=3):
        super().__init__()
        if depth < 1 or embed_dim < 1 or heads < 1 or embed_dim % heads:
            raise ValueError("Positive depth/width/heads required; width must be divisible by heads")
        self.patchify = nn.Unfold(kernel_size=4, stride=4)
        self.embedding = nn.Linear(3 * 4 * 4, embed_dim)
        self.class_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position = nn.Parameter(torch.empty(1, 65, embed_dim))
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=4 * embed_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(embed_dim, 10)
        self.apply(self._initialize)
        # TransformerEncoder 会复制初始层，单独初始化各层，避免它们从相同权重开始。
        for block in self.encoder.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, images):
        if images.shape[-2:] != (32, 32):
            raise ValueError("SmallViT expects images of size 32x32.")
        patches = self.patchify(images).transpose(1, 2)  # [batch, 64, 48]
        tokens = self.embedding(patches)  # [batch, 64, embed_dim]
        tokens = torch.cat([self.class_token.expand(len(images), -1, -1), tokens], dim=1)
        tokens = self.input_dropout(tokens + self.position)
        encoded = self.encoder(tokens)
        return encoded[:, 0]  # 分类 token 汇总整张图的信息，输出 embed_dim 维特征。

    def forward(self, images):
        return self.classifier(self.forward_features(images))


def build_model(name):
    if name == "smallcnn":
        from train import SmallCNN
        return SmallCNN()
    if name == "resnet18":
        return CifarResNet18()
    if name == "vit":
        return SmallViT()
    if name == "vit_tiny12":
        return SmallViT(depth=12, dropout=0.0)
    if name == "vit_small12":
        return SmallViT(depth=12, dropout=0.0, embed_dim=384, heads=6)
    raise ValueError(f"Unknown model: {name}")


def extract_features(model, images, model_name):
    if model_name == "smallcnn":
        x = model.features(images)
        # 返回最终分类层之前的 256 维表示。推理时 model.eval() 关闭 Dropout。
        return model.classifier[:3](x)
    return model.forward_features(images)
