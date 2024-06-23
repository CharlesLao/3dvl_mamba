import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformers import BertConfig, BertModel

from .obj_encoder import GTObjEncoder, PcdObjEncoder
from .mmt_module import MMT
from .cmt_module import CMT
from .referit3d_net import get_mlp_head, freeze_bn
from .referit3d_net import ReferIt3DNet


class ReferIt3DNetMix(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        student_model_cfg = copy.deepcopy(config)
        student_model_cfg.model_type = 'gtpcd'
        student_model_cfg.obj_encoder.use_color_enc = student_model_cfg.obj_encoder.student_use_color_enc
        self.student_model = ReferIt3DNet(student_model_cfg, device)

        
    def prepare_batch(self, batch):
        outs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                outs[key] = value.to(self.device)
            else:
                outs[key] = value
        return outs
        
    def forward(self, batch: dict, compute_loss=False, is_test=False) -> dict:
        batch = self.prepare_batch(batch)

        student_outs = self.student_model(
            batch, compute_loss=False,
            output_attentions=True, output_hidden_states=True,
        ) 
        teacher_outs = None
        if compute_loss:
            losses = self.compute_loss(teacher_outs, student_outs, batch)
            return student_outs, losses
        
        return student_outs

    def compute_loss(self, teacher_outs, student_outs, batch):
        losses = self.student_model.compute_loss(student_outs, batch)
        

        return losses

