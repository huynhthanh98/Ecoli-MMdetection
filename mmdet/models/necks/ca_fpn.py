# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from .fpn import FPN


class ContextAwareBlock(BaseModule):
    """Context-Aware Adaptive Fusion Block.
    
    This block replaces the fixed addition in standard FPN with adaptive
    fusion based on spatial and channel attention mechanisms.
    
    Args:
        channels (int): Number of input/output channels.
        reduction (int): Channel reduction ratio for channel attention.
            Defaults to 4.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 channels: int,
                 reduction: int = 4,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        
        # Spatial attention: generate spatial weight map
        self.spatial_attn = nn.Conv2d(
            channels * 2, 1, kernel_size=3, padding=1, bias=True)
        
        # Channel attention: generate channel weight vector
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Output normalization
        self.output_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
    def forward(self, x_low: Tensor, x_high: Tensor) -> Tensor:
        """Forward function.
        
        Args:
            x_low (Tensor): Lower-level feature map.
            x_high (Tensor): Higher-level feature map (will be upsampled).
            
        Returns:
            Tensor: Adaptively fused feature map.
        """
        # Step 1: Upsample high-level feature to match low-level size
        x_high_up = F.interpolate(
            x_high, size=x_low.shape[2:], mode='nearest')
        
        # Step 2: Concatenate for context modeling
        x_cat = torch.cat([x_low, x_high_up], dim=1)
        
        # Step 3: Compute attention weights
        spatial_weight = torch.sigmoid(self.spatial_attn(x_cat))
        channel_weight = self.channel_attn(x_cat)
        
        # Step 4: Adaptive fusion with spatial attention
        fused = x_low * spatial_weight + x_high_up * (1 - spatial_weight)
        
        # Step 5: Apply channel attention and output normalization
        fused = fused * channel_weight
        fused = self.output_conv(fused)
        
        return fused


@MODELS.register_module()
class CA_FPN(FPN):
    """Context-Aware Adaptive Fusion Feature Pyramid Network.
    
    This is an enhanced version of FPN that uses context-aware adaptive
    fusion instead of simple element-wise addition. Each fusion operation
    is controlled by both spatial and channel attention mechanisms.
    
    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        ca_reduction (int): Channel reduction ratio for context-aware blocks.
            Defaults to 4.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (dict, optional): Config dict for convolution layer.
        norm_cfg (dict, optional): Config dict for normalization layer.
        act_cfg (dict, optional): Config dict for activation layer.
        upsample_cfg (dict): Config dict for interpolate layer.
            Defaults to dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 ca_reduction: int = 4,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: Union[bool, str] = False,
                 relu_before_extra_convs: bool = False,
                 no_norm_on_lateral: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 upsample_cfg: ConfigType = dict(mode='nearest'),
                 init_cfg: MultiConfig = dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ) -> None:
        
        # Initialize parent FPN
        super().__init__(
            in_channels=in_channels,
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
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg
        )
        
        # Add context-aware blocks for adaptive fusion
        self.ca_blocks = nn.ModuleList()
        used_backbone_levels = self.backbone_end_level - self.start_level
        for i in range(used_backbone_levels - 1):
            ca_block = ContextAwareBlock(
                channels=out_channels,
                reduction=ca_reduction,
                init_cfg=init_cfg
            )
            self.ca_blocks.append(ca_block)
    
    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function with context-aware adaptive fusion.
        
        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.
                
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # Build laterals (same as FPN)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path with context-aware adaptive fusion
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Use context-aware block instead of simple addition
            laterals[i - 1] = self.ca_blocks[i - 1](
                laterals[i - 1], laterals[i])
        
        # Build outputs (same as FPN)
        # Part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # Part 2: add extra levels (same as FPN)
        if self.num_outs > len(outs):
            # Use max pool to get more levels on top of outputs
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # Add conv layers on top of original feature maps
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
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
