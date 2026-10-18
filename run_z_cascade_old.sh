#!/usr/bin/env bash

# Script chạy lần lượt tất cả các config trong configs/z_cascade_old
# Yêu cầu: đang ở đúng môi trường (ví dụ: conda activate test_)

set -e

CONFIG_DIR="configs/z_cascade_old"

# Thứ tự chạy các config (bạn có thể chỉnh lại nếu muốn)
  #"cascade_mask_rcnn_cbnetv2_vgg16_fpn_old.py"
  #"cascade_mask_rcnn_cbnetv2_vgg19_fpn_old.py"
  #"cascade_mask_rcnn_cbnetv2_resnet50_fpn_old.py"
CONFIGS=(
  "cascade_mask_rcnn_cbnetv2_resnet101_fpn_old.py"
  "cascade_mask_rcnn_cbnetv2_convnext_tiny_fpn_old.py"
  "cascade_mask_rcnn_cbnetv2_convnextv2_tiny_fpn_old.py"
)

  #"cascade_mask_rcnn_cbnetv2_swin_tiny_fpn_old.py"
  #"cascade_mask_rcnn_cbnetv2_swin_small_fpn_old.py"
  #"cascade_mask_rcnn_cbnetv2_vit_base_fpn_old.py"
  #"cascade_mask_rcnn_cbnetv2_pvt_tiny_fpn_old.py"
  #"cascade_mask_rcnn_cbnetv2_pvt_medium_fpn_old.py"

LOG_DIR="work_dirs/z_cascade_old_logs"
mkdir -p "$LOG_DIR"

for cfg in "${CONFIGS[@]}"; do
  echo "==================================================================="
  echo "[RUN] $CONFIG_DIR/$cfg"
  echo "Thời gian: $(date)"
  echo "Log: $LOG_DIR/${cfg%.py}.log"
  echo "==================================================================="

  python tools/train.py "$CONFIG_DIR/$cfg" \
    | tee "$LOG_DIR/${cfg%.py}.log"

done

echo "\nHoàn thành chạy tất cả config trong $CONFIG_DIR"
