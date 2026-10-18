import copy
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .cascade_rcnn import CascadeRCNN
from typing import Tuple


@MODELS.register_module()
class CascadeRCNNWithSharedAux(CascadeRCNN):
    """Cascade R-CNN with shared auxiliary supervision.
    
    This implementation follows the paper by having a parallel head for the
    assisting backbone, with shared weights for the neck and head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 aux_loss_weight: float = 0.25,
                 lead_loss_weight: float = 0.75,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.aux_loss_weight = aux_loss_weight
        self.lead_loss_weight = lead_loss_weight
        
        # Share components by direct assignment. This ensures that gradients
        # from both Loss_Lead and Loss_Assist update the same set of weights.
        self.assisting_neck = self.neck
        self.assisting_rpn_head = self.rpn_head
        self.assisting_roi_head = self.roi_head

    def forward(self, inputs: torch.Tensor, 
                data_samples: SampleList, mode: str = 'loss') -> dict:
        """Forward function."""
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        # Forward through CBNetV2 backbone (lead + assisting)
        lead_features, assisting_features = self.backbone.forward_with_assisting(
            inputs)

        # Lead backbone forward
        lead_losses = self._forward_lead(lead_features, data_samples)

        # Assisting backbone forward
        if assisting_features is not None:
            assisting_losses = self._forward_assisting(assisting_features,
                                                     data_samples)

            # Combine losses with auxiliary supervision weights
            total_losses = {}
            
            # Apply lead_loss_weight to lead losses
            for key, value in lead_losses.items():
                if isinstance(value, (list, tuple)):
                    total_losses[key] = [
                        self.lead_loss_weight * v for v in value
                    ]
                else:
                    total_losses[key] = self.lead_loss_weight * value
            
            # Apply aux_loss_weight to assisting losses
            for key, value in assisting_losses.items():
                # key is already prefixed, e.g., 'assisting_loss_bbox'
                if isinstance(value, (list, tuple)):
                    # If the loss is a sequence (e.g., from cascade stages),
                    # apply weight to each element.
                    total_losses[key] = [
                        self.aux_loss_weight * v for v in value
                    ]
                else:
                    # Otherwise, just multiply the tensor.
                    total_losses[key] = self.aux_loss_weight * value

            return total_losses
        else:
            return lead_losses

    def _forward_lead(self, features: Tuple[torch.Tensor],
                      data_samples: SampleList) -> dict:
        """Forward proposals and calculate loss for the lead head."""
        if self.neck is not None:
            features = self.neck(features)

        losses = {}
        if self.rpn_head is not None:
            proposal_cfg = self.train_cfg.get('rpn_proposal')
            rpn_data_samples = copy.deepcopy(data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                if 'labels' in data_sample.gt_instances:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                features, rpn_data_samples, proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]

        if self.roi_head is not None:
            roi_losses = self.roi_head.loss(features, rpn_results_list,
                                            data_samples)
            losses.update(roi_losses)

        return losses

    def _forward_assisting(self, features: Tuple[torch.Tensor],
                           data_samples: SampleList) -> dict:
        """Forward proposals and calculate loss for the assisting head."""
        if self.assisting_neck is not None:
            features = self.assisting_neck(features)

        losses = {}
        if self.assisting_rpn_head is not None:
            proposal_cfg = self.train_cfg.get('rpn_proposal')
            rpn_data_samples = copy.deepcopy(data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                if 'labels' in data_sample.gt_instances:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.assisting_rpn_head.loss_and_predict(
                features, rpn_data_samples, proposal_cfg=proposal_cfg)
            losses.update({f'assisting_{k}': v for k, v in rpn_losses.items()})
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]

        if self.assisting_roi_head is not None:
            roi_losses = self.assisting_roi_head.loss(features,
                                                      rpn_results_list,
                                                      data_samples)
            losses.update({f'assisting_{k}': v for k, v in roi_losses.items()})

        return losses

    def predict(self,
                inputs: torch.Tensor,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples."""
        # Only use lead backbone for inference
        lead_features, _ = self.backbone.forward_with_assisting(inputs)
        
        if self.neck is not None:
            features = self.neck(lead_features)
        else:
            features = lead_features

        # RPN forward to get proposals
        if self.rpn_head is not None:
            rpn_results_list = self.rpn_head.predict(features, data_samples)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]

        # RoIHead forward
        results_list = self.roi_head.predict(
            features, rpn_results_list, data_samples, rescale=rescale)
        
        # Add predictions to data samples
        data_samples = self.add_pred_to_datasample(data_samples,
                                                   results_list)
        return data_samples

    def _forward(self, inputs: torch.Tensor, 
                 data_samples: SampleList) -> tuple:
        """Network forward process."""
        lead_features, _ = self.backbone.forward_with_assisting(inputs)
        
        if self.neck is not None:
            features = self.neck(lead_features)
        
        return features 