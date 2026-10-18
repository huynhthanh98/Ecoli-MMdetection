import copy
from typing import Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmengine.structures import PixelData
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .hybrid_task_cascade import HybridTaskCascade


@MODELS.register_module()
class HTCWithSharedAux(HybridTaskCascade):
    """HybridTaskCascade with shared auxiliary supervision for CBNetV2.

    This detector implements the CBNetV2 auxiliary supervision strategy on top
    of the Hybrid Task Cascade (HTC) architecture.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 semantic_head: OptConfigType = None,
                 aux_loss_weight: float = 0.1,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            semantic_head=semantic_head,
            init_cfg=init_cfg)

        self.aux_loss_weight = aux_loss_weight

        # Share components by direct assignment.
        if self.with_semantic:
            self.assisting_semantic_head = self.semantic_head
        self.assisting_neck = self.neck
        if self.with_rpn:
            self.assisting_rpn_head = self.rpn_head
        self.assisting_roi_head = self.roi_head

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        lead_features, assist_features = self.backbone.forward_with_assisting(
            inputs)

        # Lead pipeline
        lead_losses = self._forward_lead(lead_features, data_samples)

        # Assisting pipeline
        if assist_features is not None:
            assisting_losses = self._forward_assisting(assist_features,
                                                     data_samples)

            # Combine losses
            total_losses = lead_losses.copy()
            for key, value in assisting_losses.items():
                # key is already prefixed, e.g., 'assisting_loss_bbox'
                if isinstance(value, (list, tuple)):
                    total_losses[key] = [
                        v * self.aux_loss_weight for v in value
                    ]
                else:
                    total_losses[key] = value * self.aux_loss_weight
            return total_losses
        else:
            return lead_losses

    def _forward_lead(self, x: Tuple[Tensor],
                      data_samples: SampleList) -> dict:
        """Forward and calculate losses for the lead pipeline."""
        losses = dict()

        # semantic segmentation branch
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_semantic = self.semantic_head.loss(semantic_pred,
                                                    data_samples)
            losses['loss_semantic_seg'] = loss_semantic
        else:
            semantic_feat = None

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal')
            rpn_data_samples = copy.deepcopy(data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                if 'labels' in data_sample.gt_instances:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)

            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]

        # roi_head forward and loss
        roi_losses = self.roi_head.loss(x, semantic_feat, rpn_results_list,
                                        data_samples)
        losses.update(roi_losses)

        return losses

    def _forward_assisting(self, x: Tuple[Tensor],
                           data_samples: SampleList) -> dict:
        """Forward and calculate losses for the assisting pipeline."""
        losses = dict()

        # semantic segmentation branch
        if self.with_semantic:
            semantic_pred, semantic_feat = self.assisting_semantic_head(x)
            loss_semantic = self.assisting_semantic_head.loss(
                semantic_pred, data_samples)
            losses['assisting_loss_semantic_seg'] = loss_semantic
        else:
            semantic_feat = None

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal')
            rpn_data_samples = copy.deepcopy(data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                if 'labels' in data_sample.gt_instances:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.assisting_rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)

            prefixed_rpn_losses = {
                f'assisting_rpn_{k}' if 'loss' in k else f'assisting_{k}': v
                for k, v in rpn_losses.items()
            }
            losses.update(prefixed_rpn_losses)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]

        # roi_head forward and loss
        roi_losses = self.assisting_roi_head.loss(x, semantic_feat,
                                                  rpn_results_list,
                                                  data_samples)
        prefixed_roi_losses = {
            f'assisting_{k}': v
            for k, v in roi_losses.items()
        }
        losses.update(prefixed_roi_losses)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples."""
        lead_features, _ = self.backbone.forward_with_assisting(inputs)
        x = self.neck(lead_features)

        # semantic segmentation branch
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        # RPN forward
        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(x, data_samples)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]

        # roi_head forward
        results_list = self.roi_head.predict(
            x,
            semantic_feat,
            rpn_results_list,
            data_samples,
            rescale=rescale)

        # Add semantic seg results
        if self.with_semantic:
            for data_sample, semantic_pred_per_img in zip(
                    data_samples, semantic_pred):
                data_sample.sem_seg = PixelData(data=semantic_pred_per_img)

        # MMDetection 3.x expects the model to return data_samples
        batch_data_samples = self.add_pred_to_datasample(
            data_samples, results_list)
        return batch_data_samples

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptMultiConfig = None) -> Tensor:
        """Network forward process. Usually not used in two-stage detectors."""
        lead_features, _ = self.backbone.forward_with_assisting(inputs)
        return self.neck(lead_features) 