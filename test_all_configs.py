#!/usr/bin/env python
"""
Script để test tất cả các file config trong thư mục z_cascade_old
Kiểm tra xem config có load được và model có khởi tạo được không
"""

import os
import sys
from pathlib import Path

# Thêm mmdetection vào path
mmdet_root = Path(__file__).parent
sys.path.insert(0, str(mmdet_root))

from mmengine.config import Config
from mmengine.registry import init_default_scope
import torch

def test_config(config_path):
    """Test một config file"""
    try:
        print(f"\n{'='*80}")
        print(f"Testing: {config_path}")
        print(f"{'='*80}")
        
        # Load config
        cfg = Config.fromfile(config_path)
        print(f"✓ Config loaded successfully")
        
        # Kiểm tra các thành phần cơ bản
        assert hasattr(cfg, 'model'), "Config không có 'model'"
        print(f"✓ Model config exists")
        
        assert hasattr(cfg, 'train_dataloader'), "Config không có 'train_dataloader'"
        print(f"✓ Train dataloader config exists")
        
        assert hasattr(cfg, 'val_dataloader'), "Config không có 'val_dataloader'"
        print(f"✓ Val dataloader config exists")
        
        # Kiểm tra backbone config
        if 'backbone' in cfg.model:
            backbone = cfg.model.backbone
            print(f"✓ Backbone type: {backbone.get('type', 'N/A')}")
            
            # Kiểm tra backbones_cfg nếu là CBNetV2
            if backbone.get('type') == 'CBNetV2':
                if 'backbones_cfg' in backbone:
                    print(f"✓ CBNetV2 có {len(backbone['backbones_cfg'])} backbones")
                    for i, bb_cfg in enumerate(backbone['backbones_cfg']):
                        print(f"  - Backbone {i+1}: {bb_cfg.get('type', 'N/A')}")
        
        # Kiểm tra neck config
        if 'neck' in cfg.model:
            neck = cfg.model.neck
            print(f"✓ Neck type: {neck.get('type', 'N/A')}")
            if 'in_channels' in neck:
                print(f"✓ Neck in_channels: {neck['in_channels']}")
        
        # Kiểm tra roi_head
        if 'roi_head' in cfg.model:
            roi_head = cfg.model.roi_head
            print(f"✓ ROI head type: {roi_head.get('type', 'N/A')}")
            if 'num_stages' in roi_head:
                print(f"✓ Num stages: {roi_head['num_stages']}")
        
        # Thử khởi tạo model (không cần data thực)
        try:
            init_default_scope('mmdet')
            from mmdet.registry import MODELS
            
            # Tạo model với dummy input size
            model_cfg = cfg.model.copy()
            
            # Khởi tạo model
            model = MODELS.build(model_cfg)
            print(f"✓ Model khởi tạo thành công: {type(model).__name__}")
            
            # Kiểm tra xem model có các thành phần cần thiết không
            if hasattr(model, 'backbone'):
                print(f"✓ Model có backbone")
            if hasattr(model, 'neck'):
                print(f"✓ Model có neck")
            if hasattr(model, 'rpn_head'):
                print(f"✓ Model có rpn_head")
            if hasattr(model, 'roi_head'):
                print(f"✓ Model có roi_head")
            
            # Kiểm tra assisting components nếu có
            if hasattr(model, 'assisting_roi_head'):
                print(f"✓ Model có assisting_roi_head")
            
            print(f"✓✓✓ Config '{config_path}' PASSED tất cả các test!")
            return True
            
        except Exception as e:
            print(f"✗ Lỗi khi khởi tạo model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗✗✗ Config '{config_path}' FAILED!")
        print(f"✗ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test tất cả các config files"""
    config_dir = mmdet_root / "configs" / "z_cascade_old"
    
    if not config_dir.exists():
        print(f"Thư mục không tồn tại: {config_dir}")
        return
    
    # Danh sách các file config cần test
    config_files = [
        "cascade_mask_rcnn_cbnetv2_convnext_tiny_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_convnextv2_tiny_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_swin_tiny_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_swin_small_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_resnet50_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_resnet101_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_vit_base_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_pvt_tiny_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_pvt_medium_fpn_old.py",
        "cascade_mask_rcnn_cbnetv2_vgg16_fpn_old.py",  # File gốc để so sánh
    ]
    
    results = {}
    total = len(config_files)
    passed = 0
    failed = 0
    
    print(f"\n{'#'*80}")
    print(f"# BẮT ĐẦU TEST {total} CONFIG FILES")
    print(f"{'#'*80}\n")
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if not config_path.exists():
            print(f"⚠ File không tồn tại: {config_path}")
            results[config_file] = False
            failed += 1
            continue
        
        success = test_config(str(config_path))
        results[config_file] = success
        if success:
            passed += 1
        else:
            failed += 1
    
    # Tóm tắt kết quả
    print(f"\n{'#'*80}")
    print(f"# TÓM TẮT KẾT QUẢ")
    print(f"{'#'*80}\n")
    print(f"Tổng số config: {total}")
    print(f"✓ PASSED: {passed}")
    print(f"✗ FAILED: {failed}")
    print(f"\n{'='*80}\n")
    
    if failed > 0:
        print("Các config FAILED:")
        for config_file, success in results.items():
            if not success:
                print(f"  ✗ {config_file}")
        print()
    
    if passed == total:
        print("🎉 TẤT CẢ CONFIG ĐỀU PASSED!")
    else:
        print(f"⚠ Có {failed} config bị lỗi, vui lòng kiểm tra lại.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)




