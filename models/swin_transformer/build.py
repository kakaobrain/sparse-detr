# ------------------------------------------------------------------------------
# Sparse DETR
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------


from collections import abc, OrderedDict
import os
import yaml

from .swin_transformer import SwinTransformer
from .config import Config

import torch


CONFIG_MAP = {
    "swin-t": "models/swin_transformer/configs/swin_tiny_patch4_window7_224.yaml",
    "swin-s": "models/swin_transformer/configs/swin_small_patch4_window7_224.yaml",
    "swin-b": "models/swin_transformer/configs/swin_base_patch4_window7_224.yaml",
    "swin-l": "models/swin_transformer/configs/swin_large_patch4_window7_224.yaml",
}


CHECKPOINT_MAP = {
    "swin-t": "/data/public/rw/team-autolearn/pretrainedmodels/swin/swin_tiny_patch4_window7_224.pth",
}


def build_model(name, out_indices, frozen_stages, pretrained):
    config_file = CONFIG_MAP[name]
    config = load_config_yaml(config_file)
    config = Config(config)
    config.freeze()
    
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(pretrain_img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                out_indices=out_indices,
                                frozen_stages=frozen_stages)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    
    if pretrained:
        ckpt_path = CHECKPOINT_MAP[name]
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict['model'], strict=False)
        
    return model


def _update_dict(tar, src):
    """recursive dict update."""
    for k, v in src.items():
        if isinstance(v, abc.Mapping):
            tar[k] = _update_dict(tar.get(k, {}), v)
        else:
            tar[k] = v
    return tar


def load_config_yaml(cfg_file, config=None):
    if config is None:
        config = OrderedDict()
    
    with open(cfg_file, 'r') as f:
        config_src = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in config_src.setdefault('BASE', ['']):
        if cfg:
            load_config_yaml(
                os.path.join(os.path.dirname(cfg_file), cfg), config
            )
    print('=> merge config from {}'.format(cfg_file))
    _update_dict(config, config_src)
    return config
