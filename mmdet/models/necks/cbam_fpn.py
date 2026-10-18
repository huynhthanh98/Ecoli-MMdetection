import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.necks.fpn import FPN
from mmdet.registry import MODELS


class CBAM(nn.Module):
    def __init__(self, channels, reduction=4, spatial_kernel=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)
        attn = self.channel_mlp(avg) + self.channel_mlp(max)
        attn = torch.sigmoid(attn).view(b, c, 1, 1)
        x = x * attn

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg, max], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_conv(spatial))
        x = x * spatial_attn

        return x


@MODELS.register_module()
class CBAMFPN(FPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam_blocks = nn.ModuleList([
            CBAM(self.out_channels) for _ in range(len(self.fpn_convs))
        ])

    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')

        outs = []
        for i in range(min(len(self.fpn_convs), len(laterals))):
            out = self.fpn_convs[i](laterals[i])
            out = self.cbam_blocks[i](out)
            outs.append(out)

        if self.num_outs > len(outs):
            orig = inputs[self.backbone_end_level - 1]
            x = outs[-1]
            for i in range(self.num_outs - len(outs)):
                if self.add_extra_convs == 'on_input':
                    x = self.fpn_convs[len(self.lateral_convs) + i](orig)
                elif self.add_extra_convs:
                    x = self.fpn_convs[len(self.lateral_convs) + i](x)
                else:
                    x = F.max_pool2d(x, 1, stride=2)
                outs.append(x)

        return tuple(outs)
