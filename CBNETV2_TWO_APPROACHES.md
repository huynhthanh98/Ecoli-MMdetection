# CBNetV2 - Hai Cách Implementation Auxiliary Supervision

## Tổng quan

Bài báo CBNetV2 đề xuất sử dụng **auxiliary supervision** để cải thiện hiệu quả training. Dựa trên đó, chúng ta có hai cách implement kiến trúc này, khác nhau ở nguồn gốc và cách áp dụng auxiliary loss.

## Cách 1: Auxiliary Supervision *tại các tầng của Cascade Head*

**File config:** `cascade_mask_rcnn_cbnetv2_r50_fpn copy.py`
**Detector:** `CascadeRCNNWithAux`

### Đặc điểm:
- **Kiến trúc:** Chỉ có một pipeline detector chính (cho lead backbone).
- **Auxiliary Supervision:** Loss phụ được tính tại các stage trung gian (stage 1, 2) của `CascadeRoIHead`. Đây là một kỹ thuật training phổ biến cho Cascade R-CNN, giúp cải thiện gradient flow cho các stage đầu. Nó không hoàn toàn giống với `Loss_Assist` trong paper.
- **Backbone:** CBNetV2 với composite connections (DHLC).
- **Neck:** FPN thông thường (vì DHLC đã thực hiện trong backbone).

### Kiến trúc:
```
Input → CBNetV2 (Lead + Assisting + DHLC) → FPN → CascadeRCNN (Auxiliary loss ở các stage trung gian)
```

## Cách 2: Auxiliary Supervision *từ Assisting Backbone* (Theo đúng paper)

**File config:** `cascade_mask_rcnn_cbnetv2_r50_fpn_shared.py`
**Detector:** `CascadeRCNNWithSharedAux`

### Đặc điểm:
- **Kiến trúc:** Implement chính xác theo hình vẽ trong paper. Assisting backbone có một pipeline detector (Neck, Head) song song với pipeline của lead backbone.
- **Auxiliary Supervision:** `Loss_Assist` được tính từ output của pipeline assisting. Total loss = `Loss_Lead` + `Loss_Assist`.
- **Shared Weights:** Các thành phần của pipeline assisting (Neck, Head) chia sẻ trọng số (share weights) với pipeline chính để tiết kiệm tài nguyên và đồng bộ hóa quá trình học.

### Kiến trúc:
```
Input → CBNetV2 (Lead + Assisting + DHLC)
  ├── Lead Features → FPN → RPN → CascadeRCNN → Loss_Lead
  └── Assisting Features → FPN (shared) → RPN (shared) → CascadeRCNN (shared) → Loss_Assist
```

## So sánh hai cách

| Tiêu chí | Cách 1 (Supervision tại Cascade Head) | Cách 2 (Supervision từ Assisting Backbone) |
|---|---|---|
| **Độ tương đồng với paper** | Ít hơn | **Chính xác theo paper** |
| **Độ phức tạp** | Đơn giản | Phức tạp hơn |
| **Memory usage** | Thấp | Cao hơn |
| **Flexibility** | Trung bình | Cao |

## Cách sử dụng

### Cách 1:
```bash
python tools/train.py configs/cascade_mask_rcnn_cbnetv2_r50_fpn\ copy.py
```

### Cách 2 (Khuyến nghị theo paper):
```bash
python tools/train.py configs/cascade_mask_rcnn_cbnetv2_r50_fpn_shared.py
```

## Kết luận

- **Cách 2** là cách implementation đúng và đầy đủ nhất theo kiến trúc và chiến lược training được đề xuất trong bài báo CBNetV2.
- **Cách 1** là một phiên bản đơn giản hóa, cũng sử dụng một dạng auxiliary supervision nhưng không hoàn toàn giống với paper.

Cả hai cách đều nhằm mục đích cải thiện hiệu năng của model. 