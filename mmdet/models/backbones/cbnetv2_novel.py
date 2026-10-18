from mmengine.model import BaseModule
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFeatureFusion(nn.Module):
    """
    🔥 NOVEL 1: Adaptive Feature Fusion with Learnable Weights
    Thay thế simple addition bằng learnable weighted fusion
    """
    def __init__(self, channels):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, lead_feat, assist_feat):
        # Concat để tính weight
        concat_feat = torch.cat([lead_feat, assist_feat], dim=1)
        weights = self.weight_net(concat_feat)  # [B, 2, 1, 1]
        
        # Adaptive weighted fusion
        fused_feat = weights[:, 0:1] * lead_feat + weights[:, 1:2] * assist_feat
        return fused_feat


class CrossScaleAttention(nn.Module):
    """
    🔥 NOVEL 2: Learnable pooling fusion between lead and assist features
    Replaces fixed pooling with learnable depthwise convs
    """
    def __init__(self, channels_list):
        super().__init__()
        self.channels_list = channels_list if isinstance(channels_list, list) else [channels_list]
        
        # Learnable spatial pooling and attention for each stage
        self.down_leads = nn.ModuleList()
        self.down_assists = nn.ModuleList()
        self.q_convs = nn.ModuleList()
        self.k_convs = nn.ModuleList()
        self.v_convs = nn.ModuleList()
        self.projs = nn.ModuleList()
        
        for channels in self.channels_list:
            # Learnable spatial pooling (depthwise conv with stride=2)
            self.down_leads.append(nn.Conv2d(channels, channels, 3, 2, 1, groups=channels))
            self.down_assists.append(nn.Conv2d(channels, channels, 3, 2, 1, groups=channels))
            
            # Lightweight attention
            self.q_convs.append(nn.Conv2d(channels, channels // 4, 1))
            self.k_convs.append(nn.Conv2d(channels, channels // 4, 1))
            self.v_convs.append(nn.Conv2d(channels, channels, 1))
            
            self.projs.append(nn.Conv2d(channels, channels, 1))
        
    def forward(self, lead_feat, assist_feat, stage_idx=0):
        """
        Args:
            lead_feat: Lead backbone feature
            assist_feat: Assisting backbone feature  
            stage_idx: Stage index for selecting conv layers
        """
        if stage_idx >= len(self.q_convs):
            stage_idx = 0
            
        # Resize assist_feat to match lead_feat spatial size
        if assist_feat.shape[-2:] != lead_feat.shape[-2:]:
            assist_feat = F.interpolate(
                assist_feat, size=lead_feat.shape[-2:], 
                mode='bilinear', align_corners=False
            )
        
        # Learnable spatial reduction
        lead_small = self.down_leads[stage_idx](lead_feat)
        assist_small = self.down_assists[stage_idx](assist_feat)
        
        # Lightweight attention
        Q = self.q_convs[stage_idx](lead_small)
        K = self.k_convs[stage_idx](assist_small)
        V = self.v_convs[stage_idx](assist_small)
        
        # Element-wise attention and fusion
        attn = torch.sigmoid(Q * K)  # element-wise attention [B, C//4, H/2, W/2]
        attn_V = attn.repeat(1, 4, 1, 1) * V  # expand attn to match V channels [B, C, H/2, W/2]
        fused = lead_small + attn_V  # fuse adaptively
        
        # Upsample and project back
        fused = F.interpolate(fused, size=lead_feat.shape[2:], mode='bilinear', align_corners=False)
        
        return self.projs[stage_idx](fused)


class ProgressiveFeatureFusion(nn.Module):
    """
    🔥 NOVEL 3: Progressive Feature Fusion
    Thay vì fusion một lúc, tiến hành progressive fusion từng bước
    """
    def __init__(self, channels):
        super().__init__()
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1)
            ) for _ in range(3)  # 3 progressive steps
        ])
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, lead_feat, assist_feats_list):
        """
        Args:
            lead_feat: Lead backbone feature ở current stage
            assist_feats_list: List of assisting features từ các high stages
        """
        result = lead_feat
        
        for i, assist_feat in enumerate(assist_feats_list):
            if i >= len(self.fusion_blocks):
                break
                
            # Resize assist_feat về same size
            assist_resized = F.interpolate(
                assist_feat, size=result.shape[-2:], 
                mode='bilinear', align_corners=False
            )
            
            # Progressive fusion
            concat_feat = torch.cat([result, assist_resized], dim=1)
            fused = self.fusion_blocks[i](concat_feat)
            
            # Gated residual connection
            gate = self.gate(fused)
            result = result + gate * fused
            
        return result


class DynamicConnectionRouter(nn.Module):
    """
    🔥 NOVEL 4: Dynamic Connection Router
    Học cách route features tối ưu dựa trên content
    """
    def __init__(self, channels, num_stages=4):
        super().__init__()
        self.num_stages = num_stages
        self.channels = channels
        
        # Router network to decide connection strength
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_stages, 1),
            nn.Sigmoid()
        )
        
        # Feature transformation for each connection
        self.transform_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stages)
        ])
        
    def forward(self, current_feat, all_assist_feats):
        """
        Args:
            current_feat: Current stage feature
            all_assist_feats: All assisting features
        """
        # Compute routing weights
        routing_weights = self.router(current_feat)  # [B, num_stages, 1, 1]
        
        result = current_feat
        for i, assist_feat in enumerate(all_assist_feats):
            if i >= self.num_stages:
                break
                
            # Transform và resize assist feature  
            assist_resized = F.interpolate(
                assist_feat, size=current_feat.shape[-2:],
                mode='bilinear', align_corners=False
            )
            transformed = self.transform_convs[i](assist_resized)
            
            # Dynamic weighted connection
            weight = routing_weights[:, i:i+1]
            result = result + weight * transformed
            
        return result


class BidirectionalFeatureExchange(nn.Module):
    """
    🔥 NOVEL 5: Bidirectional Feature Exchange
    Cho phép information flow cả 2 chiều: high↔low scales
    """
    def __init__(self, channels):
        super().__init__()
        # Upward path: low → high
        self.up_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Downward path: high → low
        self.down_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, lead_feat, assist_feat_high, assist_feat_low=None):
        """
        Args:
            lead_feat: Current lead feature
            assist_feat_high: Higher-level assist feature
            assist_feat_low: Lower-level assist feature (optional)
        """
        H, W = lead_feat.shape[-2:]
        
        # Downward flow: high → current
        down_feat = F.interpolate(
            assist_feat_high, size=(H, W), 
            mode='bilinear', align_corners=False
        )
        down_feat = self.down_conv(down_feat)
        
        features_to_fuse = [lead_feat, down_feat]
        
        # Upward flow: low → current (if available)
        if assist_feat_low is not None:
            up_feat = F.interpolate(
                assist_feat_low, size=(H, W),
                mode='bilinear', align_corners=False
            )
            up_feat = self.up_conv(up_feat)
            features_to_fuse.append(up_feat)
        else:
            # Pad với zero nếu không có low-level feature
            features_to_fuse.append(torch.zeros_like(lead_feat))
        
        # Fuse all features
        fused = torch.cat(features_to_fuse, dim=1)
        result = self.fusion_conv(fused)
        
        return result


@MODELS.register_module()
class CBNetV2Novel(BaseModule):
    """
    🚀 NOVEL CBNetV2 với 5 cải tiến chính:
    1. Adaptive Feature Fusion
    2. Cross-Scale Attention  
    3. Progressive Feature Fusion
    4. Dynamic Connection Router
    5. Bidirectional Feature Exchange
    """
    def __init__(self, backbones_cfg, fusion_type='adaptive', init_cfg=None):
        super().__init__(init_cfg)
        assert len(backbones_cfg) == 2, 'CBNetV2 cần chính xác 2 backbone'
        
        # Build backbones
        self.assisting_backbone = MODELS.build(backbones_cfg[0])
        self.lead_backbone = MODELS.build(backbones_cfg[1])
        
        # Get channels
        self.channels = self._get_backbone_channels()
        self.out_channels = self.channels
        
        # Channel alignment convs (vẫn cần cho resize)
        self.align_convs = nn.ModuleList()
        for i in range(len(self.channels)):
            convs = nn.ModuleList()
            for j in range(i+1, len(self.channels)):
                convs.append(
                    nn.Sequential(
                        nn.Conv2d(self.channels[j], self.channels[i], 1, bias=False),
                        nn.BatchNorm2d(self.channels[i]),
                        nn.ReLU(inplace=True)
                    )
                )
            self.align_convs.append(convs)
        
        # 🔥 NOVEL fusion mechanisms
        self.fusion_type = fusion_type
        
        if fusion_type == 'adaptive':
            self.fusion_modules = nn.ModuleList([
                AdaptiveFeatureFusion(channels) for channels in self.channels
            ])
        elif fusion_type == 'cross_attention':
            self.cross_attention = CrossScaleAttention(self.channels)
        elif fusion_type == 'progressive':
            self.fusion_modules = nn.ModuleList([
                ProgressiveFeatureFusion(channels) for channels in self.channels
            ])
        elif fusion_type == 'dynamic_router':
            self.fusion_modules = nn.ModuleList([
                DynamicConnectionRouter(channels, len(self.channels)) 
                for channels in self.channels
            ])
        elif fusion_type == 'bidirectional':
            self.fusion_modules = nn.ModuleList([
                BidirectionalFeatureExchange(channels) for channels in self.channels
            ])
        else:  # adaptive fusion (default)
            self.fusion_modules = nn.ModuleList([
                AdaptiveFeatureFusion(channels) for channels in self.channels
            ])

    def _get_backbone_channels(self):
        """Auto-detect backbone channels"""
        backbone = self.lead_backbone
        
        if hasattr(backbone, 'feat_dim'):
            if hasattr(backbone, 'base_channels') and hasattr(backbone, 'block'):
                base_channels = backbone.base_channels
                expansion = backbone.block.expansion
                channels = []
                for i in range(backbone.num_stages):
                    channels.append(base_channels * expansion * (2 ** i))
                return channels
        
        if hasattr(backbone, 'out_channels'):
            return backbone.out_channels
        
        # Fallback
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                feats = backbone(dummy_input)
                if isinstance(feats, (list, tuple)):
                    channels = [feat.shape[1] for feat in feats]
                else:
                    channels = [feats.shape[1]]
                return channels
        except:
            print("Warning: Không thể detect channels, dùng ResNet-50 default")
            return [256, 512, 1024, 2048]

    def forward(self, x):
        # Extract features từ cả 2 backbones
        assist_feats = self.assisting_backbone(x)
        lead_feats_raw = list(self.lead_backbone(x))
        
        # 🔥 Apply novel fusion mechanisms
        if self.fusion_type == 'cross_attention':
            # Lead-centric cross attention với assist features
            lead_feats = []
            for i, lead_feat in enumerate(lead_feats_raw):
                # Sử dụng assist feature tương ứng hoặc cuối cùng
                assist_idx = min(i, len(assist_feats) - 1)
                assist_feat = assist_feats[assist_idx]
                
                # Apply cross attention
                enhanced = self.cross_attention(lead_feat, assist_feat, stage_idx=i)
                lead_feats.append(enhanced)
        
        elif self.fusion_type == 'progressive':
            lead_feats = []
            for stage_idx, lead_feat in enumerate(lead_feats_raw):
                # Collect higher-level assist features
                higher_assist_feats = []
                for high_idx in range(stage_idx + 1, len(assist_feats)):
                    aligned = self.align_convs[stage_idx][high_idx - stage_idx - 1](
                        assist_feats[high_idx]
                    )
                    higher_assist_feats.append(aligned)
                
                # Progressive fusion
                if higher_assist_feats:
                    enhanced = self.fusion_modules[stage_idx](lead_feat, higher_assist_feats)
                else:
                    enhanced = lead_feat
                lead_feats.append(enhanced)
        
        elif self.fusion_type == 'dynamic_router':
            lead_feats = []
            for stage_idx, lead_feat in enumerate(lead_feats_raw):
                # Collect và align all assist features
                aligned_assists = []
                for assist_feat in assist_feats:
                    if assist_feat.shape[1] != lead_feat.shape[1]:
                        # Simple channel alignment
                        aligned = F.adaptive_avg_pool2d(assist_feat, 1)
                        aligned = F.interpolate(aligned, size=lead_feat.shape[-2:])
                        aligned = F.conv2d(aligned, 
                                         torch.ones(lead_feat.shape[1], assist_feat.shape[1], 1, 1).to(assist_feat.device) / assist_feat.shape[1])
                    else:
                        aligned = assist_feat
                    aligned_assists.append(aligned)
                
                # Dynamic routing
                enhanced = self.fusion_modules[stage_idx](lead_feat, aligned_assists)
                lead_feats.append(enhanced)
        
        elif self.fusion_type == 'bidirectional':
            lead_feats = []
            for stage_idx, lead_feat in enumerate(lead_feats_raw):
                # Get high-level assist feature
                if stage_idx + 1 < len(assist_feats):
                    assist_high = assist_feats[stage_idx + 1]
                else:
                    assist_high = assist_feats[-1]
                
                # Get low-level assist feature
                assist_low = assist_feats[stage_idx] if stage_idx < len(assist_feats) else None
                
                # Channel alignment
                if assist_high.shape[1] != lead_feat.shape[1]:
                    assist_high = self.align_convs[stage_idx][0](assist_high) if self.align_convs[stage_idx] else assist_high
                
                # Bidirectional exchange
                enhanced = self.fusion_modules[stage_idx](lead_feat, assist_high, assist_low)
                lead_feats.append(enhanced)
        
        else:  # adaptive fusion
            lead_feats = []
            for stage_idx, lead_feat in enumerate(lead_feats_raw):
                composite_feat = lead_feat
                
                for high_stage_idx in range(stage_idx + 1, len(assist_feats)):
                    # Align và resize
                    up_feat = F.interpolate(
                        assist_feats[high_stage_idx], 
                        size=lead_feat.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                    aligned_feat = self.align_convs[stage_idx][high_stage_idx - stage_idx - 1](up_feat)
                    
                    # 🔥 Adaptive fusion thay vì simple addition
                    composite_feat = self.fusion_modules[stage_idx](composite_feat, aligned_feat)
                
                lead_feats.append(composite_feat)
        
        return tuple(lead_feats)

    def forward_with_assisting(self, x):
        """Forward với cả lead và assisting features"""
        assist_feats = self.assisting_backbone(x)
        lead_feats = self.forward(x)
        return lead_feats, assist_feats
