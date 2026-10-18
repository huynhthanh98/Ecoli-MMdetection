import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from .cbam_fpn import CBAMFPN
from .fpn import FPN


@MODELS.register_module()
class ViTCBAMAdapter(BaseModule):
    """
    Adapter neck for ViT backbone to work with CBAM attention.
    Converts single-scale ViT features to multi-scale features and applies CBAM.
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
                 norm_cfg=None,
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
        
        # Create simple FPN first to generate multi-scale features
        self.simple_fpn = nn.ModuleDict()
        
        # Create lateral convs to convert backbone feature to different scales
        self.lateral_convs = nn.ModuleList()
        for i, in_channel in enumerate(in_channels):
            lateral_conv = ConvModule(
                backbone_channel,  # All scales use backbone_channel as input
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(lateral_conv)
        
        # Create CBAM FPN for attention mechanism
        self.cbam_fpn = CBAMFPN(
            in_channels=[out_channels] * self.num_ins,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg)
    
    def forward(self, inputs):
        """Forward function."""
        # ViT outputs single feature map
        if isinstance(inputs, (list, tuple)):
            x = inputs[-1]  # Take the last feature
        else:
            x = inputs
        
        # ViT output is usually (B, N, C) where N = H*W
        if len(x.shape) == 3:
            B, N, C = x.shape
            # Calculate H, W from patch size (assuming square)
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        elif len(x.shape) == 4 and x.shape[-1] == self.backbone_channel:
            # Format: B, H, W, C -> B, C, H, W
            x = x.permute(0, 3, 1, 2)
        elif len(x.shape) == 4:
            # Already in B, C, H, W format
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Create multi-scale features by downsampling
        features = []
        current_feat = x
        
        for i in range(self.num_ins):
            if i == 0:
                # Highest resolution (1/4 scale)
                feat = nn.functional.interpolate(
                    current_feat, scale_factor=2, mode='bilinear', align_corners=False)
            elif i == 1:
                # 1/8 scale  
                feat = current_feat
            elif i == 2:
                # 1/16 scale
                feat = nn.functional.avg_pool2d(current_feat, kernel_size=2, stride=2)
            else:
                # 1/32 scale
                feat = nn.functional.avg_pool2d(current_feat, kernel_size=4, stride=4)
            
            # Apply lateral convolution - all use backbone_channel as input
            feat = self.lateral_convs[i](feat)
            features.append(feat)
        
        # Apply CBAM FPN
        outs = self.cbam_fpn(features)
        
        return tuple(outs) 