# -*- coding: utf-8 -*-
"""
-------------------------------------------------
File Name： rollout
Description :
Author : 'li'
date： 2021/8/2
-------------------------------------------------
Change Activity:
2021/8/2:
-------------------------------------------------
"""
import torch
from torch.utils.data import DataLoader

from config.config import DEVICE, EVAL_BATCH_SIZE


def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def rollout(model, dataset):
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, DEVICE))
        return cost.data.cpu()

    return torch.cat([eval_model_bat(bat) for bat in DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)], 0)
