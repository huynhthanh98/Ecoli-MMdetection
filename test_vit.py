#!/usr/bin/env python3

import torch
import sys
import os

# Add mmdet to path
sys.path.insert(0, os.path.abspath('.'))

from mmdet.models.backbones.vit import ViT

def test_vit_backbone():
    print("Testing ViT backbone with multi-scale output...")
    
    # Create ViT model
    model = ViT(
        img_size=1024,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=dict(type='LN', requires_grad=True),
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
        use_rel_pos=True
    )
    
    # Test input
    x = torch.randn(1, 3, 1024, 1024)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        features = model(x)
    
    print(f"Number of output features: {len(features)}")
    
    for i, feat in enumerate(features):
        print(f"Feature {i+1} shape: {feat.shape}")
        print(f"Feature {i+1} min/max: {feat.min().item():.4f}/{feat.max().item():.4f}")
        print(f"Feature {i+1} mean/std: {feat.mean().item():.4f}/{feat.std().item():.4f}")
        print("---")
    
    print("✅ ViT backbone test completed!")

if __name__ == "__main__":
    test_vit_backbone()

Data Units Read:                    2,047,252 [1.04 TB]
Data Units Written:                 2,737,580 [1.40 TB]

