# -*- coding: utf-8 -*-
"""
-------------------------------------------------
File Name： train
Description :
Author : 'li'
date： 2021/8/2
-------------------------------------------------
Change Activity:
2021/8/2:
-------------------------------------------------
"""
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import LR, GRAPH_SIZE, BATCH_SIZE, DEVICE, PRE_STEP_SIZE, EPOCH_SIZE
from nets.attention_model import AttentionModel
from problems import TSP
from utils.reinforce_baselines import RolloutBaseline
from utils.rollout import move_to, set_decode_type


def _load_model_parameter(model):
    try:
        state_dict = torch.load('save/model.pth')
        model.load_state_dict(state_dict)
        print('load parameter.')
    except Exception as e:
        print('load error' + str(e))


def _train_step(model=None, baseline=None, optimizer=None, batch=None, batch_id=None):
    """

    """
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, DEVICE)
    bl_val = move_to(bl_val, DEVICE) if bl_val is not None else None
    cost, log_likelihood = model(x)
    bl_loss = 0
    if bl_val is None:
        bl_val, bl_loss = baseline.eval(x, cost)
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    # average_cost = cost.mean().cpu().numpy()
    # print('batch id :' + str(batch_id) + 'current average route distance:' + str(average_cost))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def __main():
    problem = TSP()  # 问题描述
    model = AttentionModel(problem=problem).to(DEVICE)  # 初始化模型
    _load_model_parameter(model)
    baseline = RolloutBaseline(model, problem)  # 创建基线
    optimizer = optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)
    for i in range(EPOCH_SIZE):
        print('epoch:' + str(i))
        """随机生成训练数据"""
        tmp_data = problem.make_dataset(
            size=GRAPH_SIZE, num_samples=BATCH_SIZE * PRE_STEP_SIZE, distribution=None)
        training_dataset = baseline.wrap_dataset(tmp_data)
        training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, num_workers=1)
        model.train()
        set_decode_type(model, "sampling")
        for batch_id, batch in enumerate(training_dataloader):
            """对每批次数据进行训练"""
            _train_step(model=model, baseline=baseline, optimizer=optimizer, batch=batch, batch_id=batch_id)
        lr_scheduler.step()
        torch.save(model.state_dict(), 'save/model.pth')
        baseline.epoch_callback(model, i)  # 将actor参数结果赋值给 critic
        print('save model !')


if __name__ == '__main__':
    __main()
