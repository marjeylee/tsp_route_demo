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
from torch.utils.data import DataLoader

from config.config import LR, GRAPH_SIZE, BATCH_SIZE, DEVICE, PRE_STEP_SIZE, EPOCH_SIZE
from nets.attention_model import AttentionModel
from problems import TSP
from utils.reinforce_baselines import RolloutBaseline
from utils.rollout import move_to, set_decode_type
import matplotlib.pyplot as plt


def _load_model_parameter(model):
    try:
        state_dict = torch.load('save/model.pth')
        model.load_state_dict(state_dict)
        print('load parameter.')
    except Exception as e:
        print('load error' + str(e))


def _inference(model=None, baseline=None, batch=None):
    """

    """
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, DEVICE)
    cost, log_likelihood = model(x)
    sequences, _ = model.sample_many(x)
    batch_size = x.shape[0]
    for i in range(batch_size):
        points_np = x[i, :, :]
        sequences_np = sequences[i, :]
        tmp_cost = cost[i]
        draw_route(points_np, sequences_np, tmp_cost, i)


def _sort_point(points_np, sequences_np):
    sorted_point_lst = []
    for s in sequences_np:
        p = points_np[s, :]
        sorted_point_lst.append(p)
    return sorted_point_lst


def draw_route(points_np, sequences_np, tmp_cost, step):
    """
    """
    print(tmp_cost)
    points_np = points_np.cpu()
    sequences_np = sequences_np.cpu()
    sort_point = _sort_point(points_np, sequences_np)
    for p in sort_point:
        plt.plot(p[0], p[1], "o")
    for i in range(len(sort_point) - 1):
        dx = sort_point[i + 1][0] - sort_point[i][0]
        dy = sort_point[i + 1][1] - sort_point[i][1]
        plt.quiver(sort_point[i][0], sort_point[i][1], dx, dy, angles='xy', scale=1.03, scale_units='xy', width=0.005)
    plt.savefig('C:/Users/Administrator/Desktop/img/' + str(step) + '___' + str(int(tmp_cost.min())) + '.jpg')  # 修改目标路径
    plt.clf()


def __main():
    problem = TSP()
    model = AttentionModel(problem=problem).to(DEVICE)
    _load_model_parameter(model)
    set_decode_type(model, "greedy")
    baseline = RolloutBaseline(model, problem)
    tmp_data = problem.make_dataset(
        size=GRAPH_SIZE, num_samples=100, distribution=None)
    training_dataset = baseline.wrap_dataset(tmp_data)
    training_dataloader = DataLoader(training_dataset, batch_size=100, num_workers=1)
    for batch_id, batch in enumerate(training_dataloader):
        _inference(model=model, baseline=baseline, batch=batch)


if __name__ == '__main__':
    __main()
