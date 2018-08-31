# coding: utf-8
# Author: gjwei
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable

from models.utils.layer_norm import LayerNorm

"""
Implementation of LSTM variants.
For now, they only support a sequence size of 1, and are ideal for RL use-cases. 
Besides that, they are a stripped-down version of PyTorch's RNN layers. 
(no bidirectional, no num_layers, no batch_first)
"""