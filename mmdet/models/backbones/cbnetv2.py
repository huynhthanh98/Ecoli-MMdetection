from mmengine.model import BaseModule
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import tempfile

@MODELS.register_module()
class CBNetV2(BaseModule):
    """
    CBNetV2 backbone với composite connections (DHLC - Deep High-to-Low Connection)
    Theo paper: "output of each stage in an assisting backbone flows to the parallel 
    and lower-level stages of its succeeding sibling"
    """
    def __init__(self, backbones_cfg, init_cfg=None):
        super().__init__(init_cfg)
        assert len(backbones_cfg) == 2, 'CBNetV2 cần chính xác 2 backbone (assisting, lead)'
        
        # Xử lý init_cfg: chỉ pop cho VGG, các backbone khác tự xử lý init_cfg
        assisting_cfg = copy.deepcopy(backbones_cfg[0])
        lead_cfg = copy.deepcopy(backbones_cfg[1])
        
        # Kiểm tra xem có phải VGG không
        assisting_is_vgg = assisting_cfg.get('type', '').lower() == 'vgg'
        lead_is_vgg = lead_cfg.get('type', '').lower() == 'vgg'
        
        # Chỉ pop init_cfg cho VGG (VGG không hỗ trợ init_cfg trong __init__)
        assisting_init_cfg = assisting_cfg.pop('init_cfg', None) if assisting_is_vgg else None
        lead_init_cfg = lead_cfg.pop('init_cfg', None) if lead_is_vgg else None
        
        # Tạo 2 backbone (các backbone khác sẽ tự xử lý init_cfg thông qua MODELS.build)
        self.assisting_backbone = MODELS.build(assisting_cfg)
        self.lead_backbone = MODELS.build(lead_cfg)
        
        # Load pretrained weights cho VGG nếu có (chỉ VGG cần xử lý thủ công)
        if assisting_is_vgg and assisting_init_cfg is not None:
            self._load_pretrained_for_vgg(self.assisting_backbone, assisting_init_cfg)
        if lead_is_vgg and lead_init_cfg is not None:
            self._load_pretrained_for_vgg(self.lead_backbone, lead_init_cfg)
        
        # Lấy channels từ backbone theo chuẩn MMDetection
        self.channels = self._get_backbone_channels()
        
        # Thêm thuộc tính out_channels để tương thích với neck
        self.out_channels = self.channels
        
        # Composite connections: align channels cho DHLC
        # Theo paper: feature từ assisting backbone → lower-level stages của lead backbone
        self.composite_convs = nn.ModuleList()
        for i in range(len(self.channels)):
            convs = nn.ModuleList()
            for j in range(i+1, len(self.channels)):
                # Align channel từ stage j của assisting về stage i của lead
                convs.append(
                    nn.Sequential(
                        nn.Conv2d(self.channels[j], self.channels[i], kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.channels[i]),
                        nn.ReLU(inplace=True)
                    )
                )
            self.composite_convs.append(convs)

    def _get_backbone_channels(self):
        """
        Lấy channels từ backbone theo chuẩn MMDetection
        """
        backbone = self.lead_backbone
        
        # ResNet family: có thuộc tính feat_dim để tính channels
        if hasattr(backbone, 'feat_dim'):
            # ResNet: feat_dim = block.expansion * base_channels * 2^(num_stages-1)
            # Với ResNet-50: feat_dim = 4 * 64 * 2^3 = 2048
            # Channels cho 4 stages: [256, 512, 1024, 2048]
            if hasattr(backbone, 'base_channels') and hasattr(backbone, 'block'):
                base_channels = backbone.base_channels
                expansion = backbone.block.expansion
                channels = []
                for i in range(backbone.num_stages):
                    channels.append(base_channels * expansion * (2 ** i))
                return channels
        
        # Swin Transformer: có out_channels attribute
        if hasattr(backbone, 'out_channels'):
            return backbone.out_channels
        
        # ConvNeXt: có out_channels attribute
        if hasattr(backbone, 'out_channels'):
            return backbone.out_channels
        
        # PVT: có out_channels attribute
        if hasattr(backbone, 'out_channels'):
            return backbone.out_channels
        
        # Fallback: thử với dummy input để lấy channels
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                feats = backbone(dummy_input)
                if isinstance(feats, (list, tuple)):
                    channels = [feat.shape[1] for feat in feats]
                else:
                    channels = [feat.shape[1]]
                return channels
        except:
            # Fallback cuối cùng: giả sử ResNet-50
            print("Warning: Không thể tự động detect channels, sử dụng ResNet-50 default")
            return [256, 512, 1024, 2048]

    def _load_pretrained_for_vgg(self, backbone, init_cfg):
        """Load pretrained weights cho VGG backbone"""
        if init_cfg.get('type') == 'Pretrained':
            checkpoint = init_cfg.get('checkpoint')
            prefix = init_cfg.get('prefix', '')
            if checkpoint and hasattr(backbone, 'init_weights'):
                # Xử lý URL checkpoint: download về file tạm nếu cần
                checkpoint_path = checkpoint
                is_url = checkpoint.startswith(('http://', 'https://'))
                temp_file = None
                
                if is_url:
                    # Download checkpoint từ URL về file tạm
                    filename = os.path.basename(checkpoint.split('?')[0])
                    temp_dir = tempfile.gettempdir()
                    checkpoint_path = os.path.join(temp_dir, filename)
                    if not os.path.exists(checkpoint_path):
                        # Thử nhiều URL variants nếu URL đầu tiên fail
                        urls_to_try = [checkpoint]
                        # Thử v1 nếu đang dùng v0
                        if '/v0/' in checkpoint:
                            urls_to_try.append(checkpoint.replace('/v0/', '/v1/'))
                        # Thử v1 nếu đang dùng v1
                        elif '/v1/' in checkpoint:
                            urls_to_try.append(checkpoint.replace('/v1/', '/v0/'))
                        
                        downloaded = False
                        for url in urls_to_try:
                            try:
                                print(f'Downloading checkpoint from {url} to {checkpoint_path}...')
                                torch.hub.download_url_to_file(url, checkpoint_path, progress=True)
                                downloaded = True
                                break
                            except Exception as e:
                                print(f'Warning: Failed to download from {url}: {e}')
                                if os.path.exists(checkpoint_path):
                                    os.remove(checkpoint_path)
                        
                        if not downloaded:
                            print('Warning: All checkpoint URLs failed. Skipping pretrained weights loading.')
                            return
                    temp_file = checkpoint_path
                
                # VGG.init_weights nhận pretrained path và tự load
                # Nếu có prefix, cần load và map keys thủ công
                if prefix:
                    # Load checkpoint và map keys với prefix
                    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
                    if isinstance(checkpoint_dict, dict):
                        state_dict = checkpoint_dict.get('state_dict', checkpoint_dict)
                    else:
                        state_dict = checkpoint_dict
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith(prefix):
                            new_key = k[len(prefix):]
                            new_state_dict[new_key] = v
                    if new_state_dict:
                        backbone.load_state_dict(new_state_dict, strict=False)
                else:
                    # Không có prefix, gọi init_weights trực tiếp
                    backbone.init_weights(checkpoint_path)

    def forward(self, x):
        # Assisting backbone forward
        assist_feats = self.assisting_backbone(x)
        
        # Lead backbone forward với composite connections
        lead_feats = []
        for stage_idx, feat in enumerate(self.lead_backbone(x)):
            # DHLC: tổng hợp feature từ các stage cao hơn của assisting backbone
            # Theo paper: "output of each stage in an assisting backbone flows to 
            # the parallel and lower-level stages of its succeeding sibling"
            composite_feat = feat
            for high_stage_idx in range(stage_idx + 1, len(assist_feats)):
                # Upsample feature từ stage cao của assisting về stage hiện tại của lead
                up_feat = F.interpolate(
                    assist_feats[high_stage_idx], 
                    size=feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                # Align channels từ assisting stage j về lead stage i
                aligned_feat = self.composite_convs[stage_idx][high_stage_idx - stage_idx - 1](up_feat)
                # Theo paper: feature từ assisting backbone được cộng vào lead backbone
                composite_feat = composite_feat + aligned_feat
            
            lead_feats.append(composite_feat)
        
        return tuple(lead_feats)

    def forward_with_assisting(self, x):
        """
        Forward method cho cách thứ hai: trả về cả lead và assisting features
        """
        # Assisting backbone forward
        assist_feats = self.assisting_backbone(x)
        
        # Lead backbone forward với composite connections
        lead_feats = []
        for stage_idx, feat in enumerate(self.lead_backbone(x)):
            # DHLC: tổng hợp feature từ các stage cao hơn của assisting backbone
            composite_feat = feat
            for high_stage_idx in range(stage_idx + 1, len(assist_feats)):
                # Upsample feature từ stage cao của assisting về stage hiện tại của lead
                up_feat = F.interpolate(
                    assist_feats[high_stage_idx], 
                    size=feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                # Align channels từ assisting stage j về lead stage i
                aligned_feat = self.composite_convs[stage_idx][high_stage_idx - stage_idx - 1](up_feat)
                # Theo paper: feature từ assisting backbone được cộng vào lead backbone
                composite_feat = composite_feat + aligned_feat
            
            lead_feats.append(composite_feat)
        
        return tuple(lead_feats), tuple(assist_feats) 