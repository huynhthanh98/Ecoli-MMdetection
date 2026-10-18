#!/usr/bin/env python3

import sys
import os
import torch
sys.path.insert(0, os.path.abspath('.'))

try:
    from mmdet.apis import init_detector
    config = 'configs/htc_efficientnet_b7_cbam.py'
    model = init_detector(config, device='cpu')
    print('✅ Config loaded successfully!')
    
    # Test backbone forward pass
    dummy_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        backbone_feats = model.backbone(dummy_input)
        print(f'✅ Backbone forward pass successful!')
        print(f'Number of backbone features: {len(backbone_feats)}')
        for i, feat in enumerate(backbone_feats):
            print(f'  Feature {i}: {feat.shape}')
        
        # Test neck forward pass
        neck_feats = model.neck(backbone_feats)
        print(f'✅ Neck forward pass successful!')
        print(f'Number of neck features: {len(neck_feats)}')
        for i, feat in enumerate(neck_feats):
            print(f'  Neck feature {i}: {feat.shape}')
        
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
