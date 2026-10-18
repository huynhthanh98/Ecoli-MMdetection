# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
import torch


@MODELS.register_module()
class CascadeRCNN(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
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


@MODELS.register_module()
class CascadeRCNNWithAux(CascadeRCNN):
    """
    Cascade R-CNN với auxiliary supervision cho CBNetV2
    Theo paper: auxiliary supervision được thực hiện ở các stage khác nhau của cascade
    để tăng cường training và cải thiện performance
    """
    def __init__(self, aux_loss_weight=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight

    def loss(self, batch_inputs, batch_data_samples):
        # Extract features từ CBNetV2 backbone (đã có DHLC)
        feats = self.extract_feat(batch_inputs)
        
        # Forward qua RPN và ROI head như thường
        losses = dict()
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = [d.clone() for d in batch_data_samples]
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                feats, rpn_data_samples, proposal_cfg=proposal_cfg)
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]
        
        # ROI head loss chính
        roi_losses = self.roi_head.loss(feats, rpn_results_list, batch_data_samples)
        
        # Auxiliary supervision theo paper CBNetV2
        # Theo paper: "auxiliary supervision helps to enhance the training"
        # Tăng cường supervision cho các stage trung gian để cải thiện training
        if self.training:
            # Tăng weight cho các stage trung gian (s0, s1) để tạo auxiliary effect
            # Theo paper: auxiliary supervision giúp tăng cường training
            for stage in range(len(self.roi_head.stage_loss_weights) - 1):
                for k, v in roi_losses.items():
                    if f's{stage}.' in k and 'loss' in k:
                        # Tăng weight cho auxiliary supervision theo paper
                        roi_losses[k] = v * (1 + self.aux_loss_weight)
        
        losses.update(roi_losses)
        return losses
