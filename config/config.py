# -*- coding: utf-8 -*-
"""
-------------------------------------------------
File Name： config
Description :
Author : 'li'
date： 2021/8/2
-------------------------------------------------
Change Activity:
2021/8/2:
-------------------------------------------------
"""
import torch

"""model parameter"""
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
N_ENCODE_LAYERS = 3
TANH_CLIPPING = 10.0
MASK_INNER = True
MASK_LOGITS = True
N_HEADS = 8
"""TRAIN"""
LR = 1e-5
GRAPH_SIZE = 25
BATCH_SIZE = 200
VAL_SIZE = 10
_cuda_available = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if _cuda_available else "cpu")
EVAL_BATCH_SIZE = 1000
PRE_STEP_SIZE = 30
EPOCH_SIZE = 100
