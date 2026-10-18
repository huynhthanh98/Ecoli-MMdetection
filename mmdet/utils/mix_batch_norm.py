# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class MixBN(nn.SyncBatchNorm):
    """Mix Batch Normalization for data augmentation regularization.
    
    MixBN combines multiple batch statistics during training to enhance
    regularization and improve generalization performance. It uses a 
    mixture of current batch statistics and moving average statistics.
    
    Args:
        num_features (int): Number of input features.
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-5.
        momentum (float): Momentum for the moving average. Default: 0.1.
        affine (bool): A boolean value that when set to True, the module has
            learnable affine parameters. Default: True.
        track_running_stats (bool): Track running mean/std. Default: True.
        mix_ratio (float): Ratio for mixing current and moving statistics.
            Default: 0.3.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 mix_ratio=0.3):
        super(MixBN, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats)
        self.mix_ratio = mix_ratio
        self.training_status = True  # Keep track of original training status

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f'expected 4D input (got {input.dim()}D input)')

    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input features.
            
        Returns:
            torch.Tensor: Normalized features.
        """
        self._check_input_dim(x)
        self.training_status = self.training
        
        if not self.training or not self.track_running_stats:
            # Use regular batch norm during inference
            return super().forward(x)
        
        # Save original running stats
        running_mean = self.running_mean.clone()
        running_var = self.running_var.clone()
        
        # Mix the current batch stats with running stats
        with torch.no_grad():
            # Calculate batch statistics
            batch_mean = torch.mean(x, dim=[0, 2, 3])
            batch_var = torch.var(x, dim=[0, 2, 3], unbiased=False)
            
            # Mix with running statistics
            mixed_mean = (1 - self.mix_ratio) * batch_mean + self.mix_ratio * running_mean
            mixed_var = (1 - self.mix_ratio) * batch_var + self.mix_ratio * running_var
            
            # Compute output with mixed statistics
            x_normed = (x - mixed_mean.view(1, -1, 1, 1)) / torch.sqrt(mixed_var.view(1, -1, 1, 1) + self.eps)
            
            if self.affine:
                x_normed = x_normed * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            
            # Update running stats as usual
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
        return x_normed

    def train(self, mode=True):
        """Override the train method to preserve training_status.
        
        Args:
            mode (bool): Whether to set training mode (True) or evaluation
                mode (False). Default: True.
                
        Returns:
            nn.Module: Self.
        """
        super().train(mode)
        self.training_status = mode
        return self