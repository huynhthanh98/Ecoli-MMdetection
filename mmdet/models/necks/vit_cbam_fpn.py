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
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        
        return x


@MODELS.register_module()
class ViTCBAMFPN(BaseModule):
    """
    CBAM FPN designed specifically for ViT backbone.
    Handles single ViT output and creates multi-scale features with CBAM attention.
    """
    
    def __init__(self,
                 in_channels=768,
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs='on_input',
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ViTCBAMFPN, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.add_extra_convs = add_extra_convs
        
        # Reduce channels from ViT output to FPN channels
        self.reduce_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if not no_norm_on_lateral else None,
            act_cfg=act_cfg,
            inplace=False)
        
        # CBAM attention on reduced feature
        self.cbam = CBAM(out_channels)
        
        # FPN convs for different scales
        self.fpn_convs = nn.ModuleList()
        for i in range(num_outs):
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
        
        # Extra convs for additional scales
        if add_extra_convs:
            extra_levels = num_outs - 4  # Assume 4 natural levels (P2, P3, P4, P5)
            if extra_levels > 0:
                self.extra_convs = nn.ModuleList()
                for i in range(extra_levels):
                    if add_extra_convs == 'on_input':
                        extra_conv = ConvModule(
                            in_channels,  # Use original ViT channels
                            out_channels,
                            3,
                            stride=2,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            inplace=False)
                    else:
                        extra_conv = ConvModule(
                            out_channels,
                            out_channels,
                            3,
                            stride=2,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            inplace=False)
                    self.extra_convs.append(extra_conv)
    
    def forward(self, inputs):
        """Forward function."""
        # ViT outputs single feature map
        if isinstance(inputs, (list, tuple)):
            x = inputs[-1]  # Take the last feature
        else:
            x = inputs
        
        # Handle different ViT output formats
        if len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        elif len(x.shape) == 4 and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2)
        
        # Store original input for extra convs
        orig_input = x
        
        # Reduce channels and apply CBAM
        x = self.reduce_conv(x)
        x = self.cbam(x)
        
        # Create multi-scale features from single input
        outs = []
        
        # P2 (1/4): Upsample x2
        p2 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.fpn_convs[0](p2)
        outs.append(p2)
        
        # P3 (1/8): Original scale
        p3 = self.fpn_convs[1](x)
        outs.append(p3)
        
        # P4 (1/16): Downsample /2 (only if large enough)
        if x.shape[2] >= 2 and x.shape[3] >= 2:
            p4 = F.avg_pool2d(x, kernel_size=2, stride=2)
        else:
            p4 = x  # Use original if too small
        p4 = self.fpn_convs[2](p4)
        outs.append(p4)
        
        # P5 (1/32): Downsample /4 or /2 from p4
        if x.shape[2] >= 4 and x.shape[3] >= 4:
            p5 = F.avg_pool2d(x, kernel_size=4, stride=4)
        elif p4.shape[2] >= 2 and p4.shape[3] >= 2:
            p5 = F.avg_pool2d(p4, kernel_size=2, stride=2)
        else:
            p5 = p4  # Use p4 if too small
        p5 = self.fpn_convs[3](p5)
        outs.append(p5)
        
        # Additional scales if needed (P6, P7, ...)
        if self.num_outs > 4:
            if self.add_extra_convs == 'on_input':
                # Use original ViT input for extra scales
                extra_source = orig_input
                for i in range(self.num_outs - 4):
                    if hasattr(self, 'extra_convs'):
                        extra_out = self.extra_convs[i](extra_source)
                        outs.append(extra_out)
                        extra_source = extra_out  # Chain for next level
            else:
                # Use last FPN output for extra scales
                for i in range(self.num_outs - 4):
                    if self.relu_before_extra_convs:
                        extra_out = F.relu(outs[-1])
                    else:
                        extra_out = outs[-1]
                    
                    if hasattr(self, 'extra_convs'):
                        extra_out = self.extra_convs[i](extra_out)
                    else:
                        extra_out = F.max_pool2d(extra_out, 1, stride=2)
                    outs.append(extra_out)
        
        return tuple(outs)
