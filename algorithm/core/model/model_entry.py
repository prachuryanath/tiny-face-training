import torch
from core.utils.config import configs 
from quantize.custom_quantized_format import build_quantized_network_from_cfg
from quantize.quantize_helper import create_scaled_head, create_quantized_head
from ..ofa_nn.networks.mobilenet_v2 import MobileNetV2

__all__ = ['build_mcu_model']


# def build_mcu_model():
#     cfg_path = f"assets/mcu_models/{configs.net_config.net_name}.pkl"
#     cfg = torch.load(cfg_path, weights_only=False)
    
#     model = build_quantized_network_from_cfg(cfg, n_bit=8)

#     if configs.net_config.mcu_head_type == 'quantized':
#         model = create_quantized_head(model)
#     elif configs.net_config.mcu_head_type == 'fp':
#         model = create_scaled_head(model, norm_feat=False)
#     else:
#         raise NotImplementedError

#     return model

def build_mcu_model():
    # PHASE 1 CONFIGURATION
    # Initialize standard MobileNetV2 from your library
    # Ensure 'num_classes' is set to your embedding dimension (e.g., 128) effectively making the "classifier" the embedding layer
    embedding_dim = configs.net_config.get('embedding_dim', 128) 
    
    model = MobileNetV2(
        n_classes=embedding_dim, 
        width_mult=configs.net_config.get('width_mult', 0.35),
        dropout_rate=0.0
    )
    return model