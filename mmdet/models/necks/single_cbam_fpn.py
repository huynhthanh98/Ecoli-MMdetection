import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS


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
class SingleCBAMFPN(BaseModule):
    """
    CBAM FPN for single-scale ViT backbone.
    Takes single ViT output and creates multi-scale features with CBAM attention.
    """
    
    def __init__(self,
                 in_channels=768,
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs='on_input',
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SingleCBAMFPN, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        
        # Initial conv to reduce channels
        self.reduce_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        
        # CBAM attention on the main feature
        self.cbam = CBAM(out_channels)
        
        # Convs for different scales
        self.scale_convs = nn.ModuleList()
        for i in range(num_outs):
            scale_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.scale_convs.append(scale_conv)
        
        # Extra convs for additional scales
        if add_extra_convs:
            self.extra_convs = nn.ModuleList()
            for i in range(max(0, num_outs - 4)):  # Assume we have 4 natural scales
                extra_conv = ConvModule(
                    in_channels if add_extra_convs == 'on_input' else out_channels,
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
            x = inputs[-1]
        else:
            x = inputs
        
        # Handle different ViT output formats
        if len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        elif len(x.shape) == 4 and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2)
        
        # Reduce channels and apply CBAM
        x = self.reduce_conv(x)
        x = self.cbam(x)  # Apply CBAM attention to the main feature
        
        # Create multi-scale features from single input
        outs = []
        
        # P2 (1/4): Upsample x2
        p2 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.scale_convs[0](p2)
        outs.append(p2)
        
        # P3 (1/8): Original scale
        p3 = self.scale_convs[1](x)
        outs.append(p3)
        
        # P4 (1/16): Downsample /2
        p4 = F.avg_pool2d(x, kernel_size=2, stride=2)
        p4 = self.scale_convs[2](p4)
        outs.append(p4)
        
        # P5 (1/32): Downsample /4 from original x, or /2 from p4
        if x.shape[2] >= 4 and x.shape[3] >= 4:
            p5 = F.avg_pool2d(x, kernel_size=4, stride=4)
        else:
            # If x is too small, downsample from p4
            p5 = F.avg_pool2d(p4, kernel_size=2, stride=2)
        p5 = self.scale_convs[3](p5)
        outs.append(p5)
        
        # Additional scales if needed
        if self.num_outs > 4:
            if self.add_extra_convs == 'on_input':
                # Use original input for extra scales
                orig_input = inputs[-1] if isinstance(inputs, (list, tuple)) else inputs
                if len(orig_input.shape) == 3:
                    B, N, C = orig_input.shape
                    H = W = int(N ** 0.5)
                    extra_source = orig_input.transpose(1, 2).reshape(B, C, H, W)
                elif len(orig_input.shape) == 4 and orig_input.shape[-1] == self.in_channels:
                    extra_source = orig_input.permute(0, 3, 1, 2)
                else:
                    extra_source = orig_input
                    
                # P6: Extra downsample
                if hasattr(self, 'extra_convs'):
                    extra_out = self.extra_convs[0](extra_source)
                    outs.append(extra_out)
            else:
                # Use last output for extra scales
                for i in range(self.num_outs - 4):
                    if self.relu_before_extra_convs:
                        extra_out = F.relu(outs[-1])
                    else:
                        extra_out = outs[-1]
                    extra_out = F.max_pool2d(extra_out, kernel_size=2, stride=2)
                    outs.append(extra_out)
        
        return tuple(outs) 