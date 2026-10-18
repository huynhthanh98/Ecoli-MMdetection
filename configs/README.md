# CBNetV2 Implementations in MMDetection

This directory contains configuration files for the CBNetV2 architecture, implementing two different strategies for auxiliary supervision as interpreted from the original paper ([arXiv:2107.00420](https://arxiv.org/pdf/2107.00420)).

---

## Method 1: Efficient Auxiliary Supervision (Text-based)

This approach is based on the description in **Section 3.2** of the paper, which is a practical and computationally efficient interpretation.

- **Concept:** It uses a single, standard detector pipeline (`Neck` + `CascadeRoIHead`). "Auxiliary supervision" is achieved by increasing the loss weights for the intermediate stages (e.g., stage 0 and 1) of the `CascadeRoIHead`, forcing them to learn better representations earlier.
- **Pros:**
    - ✅ Computationally efficient (only one Neck and Head forward pass).
    - ✅ Simple and clean integration with standard MMDetection components.
- **Cons:**
    - ❌ The assisting backbone is only trained indirectly via gradients flowing back from the lead backbone.
- **Detector Used:** `CascadeRCNNWithAux`
- **How to Run:**
  ```bash
  # Train with ResNet-50
  python tools/train.py configs/cbnetv2/cascade_mask_rcnn_cbnetv2_r50_fpn.py
  ```

---

## Method 2: Figure-based Auxiliary Supervision (Diagram-based)

This approach strictly follows the architecture shown in **Figure 4** of the paper. It is more faithful to the diagram but more resource-intensive.

- **Concept:** It uses two complete, parallel detector pipelines. The assisting backbone's features are fed into a full Neck and Head to calculate a direct `Loss_Assist`. The `Neck` and `CascadeRoIHead` modules **share weights** between the two pipelines.
- **Pros:**
    - ✅ Directly trains the assisting backbone with its own detection task, which might lead to better specialized features.
    - ✅ A 1-to-1 implementation of the paper's diagram.
- **Cons:**
    - ❌ Significantly more resource-intensive, as it nearly doubles the computation for the detector's Neck and Head.
- **Detector Used:** `CascadeRCNNCBNetV2Fig4`
- **How to Run:**
  ```bash
  # Train with ResNet-50
  python tools/train.py configs/cbnetv2/cascade_mask_rcnn_cbnetv2_fig4_r50_fpn.py
  ```

---

## Which one to choose?

-   **Start with Method 1** for a great balance of performance and training efficiency.
-   **Use Method 2** if you have ample computational resources and want to experiment with an implementation that is more faithful to the paper's diagram. 