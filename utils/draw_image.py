# -*- coding: utf-8 -*-
"""
-------------------------------------------------
File Name： draw_image
Description :
Author : 'li'
date： 2021/7/29
-------------------------------------------------
Change Activity:
2021/7/29:
-------------------------------------------------
"""
import matplotlib.pyplot as plt


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
    plt.savefig('C:/Users/Administrator/Desktop/imgs/20/' + str(step) + '___' + str(int(tmp_cost.min())) + '.jpg')
    plt.clf()


if __name__ == '__main__':
    draw_route(1, 1)
