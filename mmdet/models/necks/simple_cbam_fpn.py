import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from .fpn import FPN


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        
        return x


@MODELS.register_module()
class SimpleCBAMFPN(BaseModule):
    """
    Simple FPN with CBAM attention for ViT backbone.
    Based on SimpleFPN but adds lightweight CBAM attention.
    """
    
    def __init__(self,
                 backbone_channel=768,
                 in_channels=[192, 384, 768, 768],
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN2d', requires_grad=True),
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        # Build lateral convs
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                backbone_channel,  # All use backbone_channel as input
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
        
        # Build FPN convs
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
        
        # Add CBAM attention modules
        self.cbam_modules = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            cbam = CBAM(out_channels)
            self.cbam_modules.append(cbam)
        
        # Add extra convs
        if self.add_extra_convs:
            for i in range(self.num_outs - self.backbone_end_level + self.start_level):
                if self.add_extra_convs == 'on_input':
                    extra_fpn_conv = ConvModule(
                        backbone_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                else:
                    extra_fpn_conv = ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
    
    def forward(self, inputs):
        """Forward function."""
        # ViT outputs single feature map
        if isinstance(inputs, (list, tuple)):
            x = inputs[-1]
        else:
            x = inputs
        
        # Handle different ViT output formats
        if len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        elif len(x.shape) == 4 and x.shape[-1] == self.backbone_channel:
            x = x.permute(0, 3, 1, 2)
        
        # Generate multi-scale features
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i == 0:
                # P2: upsample
                feat = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            elif i == 1:
                # P3: original
                feat = x
            elif i == 2:
                # P4: downsample
                feat = F.avg_pool2d(x, kernel_size=2, stride=2)
            else:
                # P5: downsample more
                feat = F.avg_pool2d(x, kernel_size=4, stride=4)
            
            lateral = lateral_conv(feat)
            laterals.append(lateral)
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale_factor` (e.g. 2) is preferred, but
            # it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        
        # Build outputs with FPN convs and CBAM attention
        outs = []
        for i in range(used_backbone_levels):
            out = self.fpn_convs[i](laterals[i])
            # Apply CBAM attention
            out = self.cbam_modules[i](out)
            outs.append(out)
        
        # Add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = x
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        
        return tuple(outs) 