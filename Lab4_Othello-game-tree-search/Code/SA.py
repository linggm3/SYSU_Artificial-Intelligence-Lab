from Class import Chess
import math
import random
import numpy as np


def new_weight_generate(old_weight):
    n_weight = old_weight
    for i in range(4):  # 对每个参数
        for j in range(i+1):   # 对每个参数
            if np.random.rand() < 0.2:
                n_weight[i][j] += random.randint(-50, 50)  # 扰动
    if np.random.rand() < 0.2:
        n_weight[4] += random.randint(-50, 50)  # 扰动
    if np.random.rand() < 0.2:
        n_weight[5] += random.randint(-50, 50)  # 扰动
    return n_weight


if __name__ == '__main__':
    T = 100
    inner_loop = 10
    weight = [[704.1267941755075], [-42.96383722364534, -167.91415998636808], [183.1490661265754, -117.29640043233871, -49.99951016725327], [41.904438956974744, -61.67663752217178, -9.123413414163284, -29.476952051187183], 173.17347724054167, 25.318793643976004]
    while T > 0.01:
        for loop in range(inner_loop):
            new_weight = new_weight_generate(weight)
            chess1 = Chess(new_weight)  # 新解
            chess2 = Chess(weight)  # 旧解
            for epoch in range(30):  # 对战
                [res, pos] = chess1.minimax_search_with_abcut(epoch, epoch + 3, 1, show=False)
                if len(pos) != 0:  # 无子可下
                    chess1.drops(pos[0], pos[1], 1)
                    chess2.drops(pos[0], pos[1], 1)

                [res, pos] = chess2.minimax_search_with_abcut(epoch, epoch + 3, -1, show=False)
                if len(pos) != 0:  # 无子可下
                    chess1.drops(pos[0], pos[1], -1)
                    chess2.drops(pos[0], pos[1], -1)

            final_score = chess1.final_score()  # 最终得分
            if final_score > 0:  # 新解比旧解好，接受新解
                weight = new_weight
            elif math.exp(-final_score / T > np.random.rand()):  # 新解比旧解差，以一定概率接受新解
                weight = new_weight
            print('T:', T, ' inner_loop:', loop)
            print('score:', final_score/10000)
            print('weight:', weight, '\n')
        T *= 0.98  # 退温
