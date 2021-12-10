# ------------------------------------------------------------------------------------
# Sparse DETR
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------


"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from models import swin_transformer
from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rsqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, args):
        # TODO: args -> duplicated args
        super().__init__()
        if 'none' in args.backbone:
            self.strides = [1]  # not used, actually (length only matters)  
            self.num_channels = [3]
            return_layers = self.get_return_layers('identity', (0,))
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        elif 'resnet' in args.backbone:
            
            if not args.backbone_from_scratch and not args.finetune_early_layers:
                print("Freeze early layers.")
                for name, parameter in backbone.named_parameters():
                    if not train_backbone or all([k not in name for k in ['layer2', 'layer3', 'layer4']]):
                        parameter.requires_grad_(False)
            else:
                print('Finetune early layers as well.')
                    
            layer_name = "layer"
            if return_interm_layers:
                return_layers = self.get_return_layers(layer_name, (2, 3, 4))
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
            else:
                return_layers = self.get_return_layers(layer_name, (4,))
                self.strides = [32]
                self.num_channels = [2048]
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
                
        elif 'swin' in args.backbone:
            if return_interm_layers:
                num_channels = [int(backbone.embed_dim * 2 ** i) for i in range(backbone.num_layers)]
                return_layers = [2, 3, 4]
                self.strides = [8, 16, 32]
                self.num_channels = num_channels[1:]
            else:
                return_layers = [4]
                self.strides = [32]
                self.num_channels = num_channels[-1]
            self.body = backbone
                
        else:
            raise ValueError(f"Unknown backbone name: {args.backbone}")
        
    @staticmethod
    def get_return_layers(name: str, layer_ids):
        return {name + str(n): str(i) for i, n in enumerate(layer_ids)}

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    
    
class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.identity0 = torch.nn.Identity()


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 args):
        print(f"Backbone: {name}")
        pretrained = is_main_process() and not args.backbone_from_scratch and not args.scrl_pretrained_path
        if not pretrained:
            print("Train backbone from scratch.")
        else:
            print("Load pretrained weights")
        
        if "none" in name:
            backbone = DummyBackbone()
        elif "resnet" in name:
            assert name not in ("resnet18", "resnet34"), "number of channels are hard coded"
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
        elif "swin" in name:
            assert not dilation, "not supported"
            if not args.backbone_from_scratch and not args.finetune_early_layers:
                print("Freeze early layers.")
                frozen_stages = 2
            else:
                print('Finetune early layers as well.')
                frozen_stages = -1
            if return_interm_layers:
                out_indices = [1, 2, 3]
            else:
                out_indices = [3]
                
            backbone = swin_transformer.build_model(
                name, out_indices=out_indices, frozen_stages=frozen_stages, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone name: {args.backbone}")
            
        if args.scrl_pretrained_path:
            assert "resnet" in name, "Currently only resnet50 is available."
            ckpt = torch.load(args.scrl_pretrained_path, map_location="cpu")
            translate_map = {
                "encoder.0" : "conv1",
                "encoder.1" : "bn1",
                "encoder.4" : "layer1",
                "encoder.5" : "layer2",
                "encoder.6" : "layer3",
                "encoder.7" : "layer4",
            }
            state_dict = {
                translate_map[k[:9]] + k[9:] : v
                for k, v in ckpt["online_network_state_dict"].items()
                if "encoder" in k
            }
            backbone.load_state_dict(state_dict, strict=False)
        
        super().__init__(backbone, train_backbone, return_interm_layers, args)
        if dilation and "resnet" in name:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
    
    
def test_backbone(backbone):
    imgs = [
        torch.randn(2, 3, 633, 122),
        torch.randn(2, 3, 322, 532),
        torch.randn(2, 3, 236, 42),
    ]
    return [backbone(img).shape for img in imgs]


def build_backbone(args):
    # test_backbone(torchvision.models.resnet50())
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args)
    model = Joiner(backbone, position_embedding)
    return model
