from mmdet.models.necks.fpn import FPN
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class CBNetFPN(FPN):
    """
    FPN cho CBNetV2: chỉ là FPN thông thường vì DHLC đã được thực hiện trong backbone
    """
    def __init__(self, in_channels, out_channels, num_outs, **kwargs):
        super().__init__(in_channels, out_channels, num_outs, **kwargs)

    def forward(self, inputs):
        # inputs: feature maps từ CBNetV2 backbone (đã có DHLC)
        return super().forward(inputs) 