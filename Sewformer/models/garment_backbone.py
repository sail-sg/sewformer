import torch
import torch.nn.functional as F
from torch import nn

import sys
from .utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .garment_detr_2d import MLP, SetCriterionWithOutMatcher

from metrics.losses import *

class GarmentBackbone(nn.Module):
    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.pooling = nn.AdaptiveMaxPool1d(num_classes)
        self.hidden_dim = 256
        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(self.hidden_dim, 1)
        self.panel_embed = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 1)
        self.panel_decoder = MLP(self.hidden_dim, self.hidden_dim, 56, 2)
        self.panel_rt_decoder = MLP(self.hidden_dim, self.hidden_dim, 7, 2)
    
    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        
        src, mask = features[-1].decompose()
        bs = src.shape[0]
        assert mask is not None
        hs = self.pooling(self.input_proj(src).view(bs, self.hidden_dim, -1).contiguous()).transpose(1, 2).contiguous()
        output_class = self.class_embed(hs)
        output_panel_embed = self.panel_embed(hs)
        output_panel = self.panel_decoder(output_panel_embed)
        output_panel_rt = self.panel_rt_decoder(output_panel_embed)
        output_rotations = output_panel_rt[:, :, :4]
        output_translations = output_panel_rt[:, :, 4:]
        out = {"output_class": output_class, 
               "outlines": output_panel, 
               "rotations": output_rotations, 
               "translations": output_translations}
        return out 

def build(args):
    num_classes = args["dataset"]["max_pattern_len"]
    devices = torch.device(args["trainer"]["devices"][0] if isinstance(args["trainer"]["devices"], list) else args["trainer"]["devices"])
    backbone = build_backbone(args)
    
    model = GarmentBackbone(backbone, num_classes)
    criterion = SetCriterionWithOutMatcher(args["dataset"], args["NN"]["loss"])
    return model, criterion