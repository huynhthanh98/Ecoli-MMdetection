import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule # ConvModule có thể không cần trực tiếp ở đây nếu FPN gốc xử lý
from mmdet.models.necks.fpn import FPN
from mmdet.registry import MODELS

# --- SE Module ---
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# --- CBAM Components ---
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Sử dụng Conv2d với kernel 1x1 thay cho Linear để giữ tính tương thích không gian nếu cần
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMModule(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x_ca_map = self.ca(x)
        x_after_ca = x_ca_map * x
        x_sa_map = self.sa(x_after_ca)
        x_out = x_sa_map * x_after_ca
        return x_out

# --- GatedSECBAM Module ---
class GatedSECBAM(nn.Module):
    def __init__(self, channels, se_reduction=16, cbam_reduction=16, cbam_spatial_kernel=7, gate_reduction=4):
        super().__init__()
        self.se_module = SEModule(channels, se_reduction)
        self.cbam_module = CBAMModule(channels, cbam_reduction, cbam_spatial_kernel)

        self.avg_pool_gate = nn.AdaptiveAvgPool2d(1)
        self.gate_fc = nn.Sequential(
            nn.Linear(channels, channels // gate_reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // gate_reduction, 2, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        se_attended = self.se_module(x)
        cbam_attended = self.cbam_module(x)

        gate_input = self.avg_pool_gate(x).view(b, c)
        gate_logits = self.gate_fc(gate_input)
        gate_weights = F.softmax(gate_logits, dim=1)

        w_se = gate_weights[:, 0].view(b, 1, 1, 1)
        w_cbam = gate_weights[:, 1].view(b, 1, 1, 1)

        out = w_se * se_attended + w_cbam * cbam_attended
        return out

# --- GatedSECBAMFPN ---
@MODELS.register_module()
class GatedSECBAMFPN(FPN):
    def __init__(self,
                 *args,
                 se_reduction=16,
                 cbam_reduction=16,
                 cbam_spatial_kernel=7,
                 gate_reduction=4,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Số lượng FPN convs chính (không tính extra_convs)
        # self.num_inputs là số lượng lateral connections
        num_main_fpn_levels = len(self.in_channels)
        
        self.gated_attention_blocks = nn.ModuleList()
        for _ in range(num_main_fpn_levels):
            self.gated_attention_blocks.append(
                GatedSECBAM(channels=self.out_channels,
                              se_reduction=se_reduction,
                              cbam_reduction=cbam_reduction, # Thêm tham số này
                              cbam_spatial_kernel=cbam_spatial_kernel,
                              gate_reduction=gate_reduction)
            )
        # In ra để kiểm tra
        print(f"Initialized GatedSECBAMFPN with {len(self.gated_attention_blocks)} attention blocks.")
        print(f"Number of fpn_convs in FPN superclass: {len(self.fpn_convs)}")
        print(f"Number of lateral_convs in FPN superclass: {len(self.lateral_convs)}")
        print(f"Number of input levels: {len(self.in_channels)}")


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, ``scale_factor`` is needed here
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # Build outputs
        # Part 1: from original levels
        outs = []
        for i in range(used_backbone_levels):
            fpn_conv_out = self.fpn_convs[i](laterals[i])
            # Apply GatedSECBAM only to the main FPN levels
            if i < len(self.gated_attention_blocks): # Đảm bảo không vượt quá số block đã tạo
                 gated_out = self.gated_attention_blocks[i](fpn_conv_out)
                 outs.append(gated_out)
            else: # Nếu có fpn_convs nhiều hơn số gated_attention_blocks (ví dụ cho P6 do add_extra_convs='on_output')
                 outs.append(fpn_conv_out)


        # Part 2: add extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                
                if self.relu_before_extra_convs:
                    extra_source = F.relu(extra_source)
                
                outs.append(self.fpn_convs[used_backbone_levels](extra_source)) # fpn_convs[used_backbone_levels] là extra_conv đầu tiên
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs) 