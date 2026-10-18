import torch
import torch.nn as nn
from mmdet.models.necks import FPN
from mmcv.cnn import build_norm_layer
from mmdet.registry import MODELS  # ← Thêm dòng này để đăng ký module

@MODELS.register_module()  # ← Decorator này sẽ đăng ký lớp vào registry
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

@MODELS.register_module()  # ← Đăng ký FPNWithSE vào registry của MMDetection
class FPNWithSE(FPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.se_blocks = nn.ModuleList([
            SELayer(self.out_channels) for _ in range(self.num_outs)
        ])

    def forward(self, inputs):
        outs = super().forward(inputs)
        outs = [se(out) for out, se in zip(outs, self.se_blocks)]
        return tuple(outs)
