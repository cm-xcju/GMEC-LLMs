# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: the definition of A wrapper of MCAN for finetuning with the 
# strategy described in the paper.
# ------------------------------------------------------------------------------ #

import torch
from torch import nn
from torch.nn import functional as F

from .mecpe import *
from .gpt2 import *


class MECPEForTrain(GPT2LMHeadModel):
    """
    A wrapper of MCAN for finetuning with the strategy described 
    in the paper. We inherit the parameters of existing answers 
    and append new parameters for the new answers.
    """
    def __init__(self, __C,config):
        self.__C=__C
        stop()
        super().__init__(__C.BERT_VERSION,config)

        self.proj1 = nn.Linear(__C.FLAT_OUT_SIZE,1)

    @torch.no_grad()
    def parameter_init(self):
        self.proj1.weight.data.zero_()
        self.proj1.bias.data = self.proj.bias.data.mean() + torch.zeros(self.proj1.bias.data.shape)

    def forward(self, input_tuple, output_answer_latent=False):
        proj_feat, answer_latent = super().forward(input_tuple, output_answer_latent=True)
        proj_feat = torch.cat([
            proj_feat,
            self.proj1(answer_latent)
        ], dim=1)
        
        if output_answer_latent:
            return proj_feat, answer_latent

        return proj_feat
