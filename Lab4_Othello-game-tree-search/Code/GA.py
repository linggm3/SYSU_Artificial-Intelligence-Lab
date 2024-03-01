from Class import Chess
import math
import numpy as np
import random
import copy
import time
import multiprocessing


# 交叉
def cross_over(pre_pop, p=0.75):
    arr1 = np.array(range(0, len(pre_pop)))
    arr2 = np.array(range(0, len(pre_pop)))
    np.random.shuffle(arr1)
    np.random.shuffle(arr2)
    cross_overed_pop = copy.deepcopy(pre_pop)
    for e in range(len(pre_pop)):
        fa1 = pre_pop[arr1[e]]
        fa2 = pre_pop[arr2[e]]
        child = copy.deepcopy(fa1)
        for i in range(4):
            for j in range(1 + i):
                if np.random.rand() < p:
                    child[i][j] = (fa1[i][j] + fa2[i][j]) / 2 + 1.5 * abs(fa1[i][j] - fa2[i][j]) * (np.random.rand() - 0.5)
        if np.random.rand() < p:    
            child[4] = (fa1[4] + fa2[4]) / 2 + 1.5 * abs(fa1[4] - fa2[4]) * (np.random.rand() - 0.5)
        if np.random.rand() < p:
            child[5] = (fa1[5] + fa2[5]) / 2 + 1.5 * abs(fa1[5] - fa2[5]) * (np.random.rand() - 0.5)
        cross_overed_pop.append(child)
    return cross_overed_pop


# 变异
def mutation(pre_pop, p=0.1, r=50):
    # 对每个策略
    for e in range(int(len(pre_pop)/2), len(pre_pop)):
        for i in range(4):
            for j in range(1+i):
                if np.random.rand() < p:
                    pre_pop[e][i][j] += random.randint(-r, r)
        if np.random.rand() < p:
            pre_pop[e][4] += random.randint(-r, r)
        if np.random.rand() < p:
            pre_pop[e][5] += random.randint(-r, r)
    return pre_pop


# 锦标赛
def selection(pre_pop):
    size = len(pre_pop)
    now_pop = []
    arr = np.array(range(0, size))
    np.random.shuffle(arr)
    process = []
    que = multiprocessing.Queue()
    for i in range(int(size/2)):
        process.append(multiprocessing.Process(target=race, args=(pre_pop[i], pre_pop[int(size/2)+i], que)))
        process[i].start()
    for i in range(int(size / 2)):
        process[i].join()
    now_pop = [que.get() for p in process]
    return now_pop


# 两个 weight 对打
def race(weight1, weight2, que):
    chess1 = Chess(weight1)
    chess2 = Chess(weight2)
    for step in range(30):
        [res, pos] = chess1.minimax_search_with_abcut(step, step + 3, 1, show=False)
        if len(pos) != 0:
            chess1.drops(pos[0], pos[1], 1)
            chess2.drops(pos[0], pos[1], 1)
        # chess1.show_board(0.5, title='score: ' + str(chess1.final_score()))

        [res, pos] = chess2.minimax_search_with_abcut(step, step + 3, -1, show=False)
        if len(pos) != 0:
            chess1.drops(pos[0], pos[1], -1)
            chess2.drops(pos[0], pos[1], -1)
        # chess1.show_board(0.5, title='score: ' + str(chess1.final_score()))
    final_score = chess1.final_score()
    # 赢家加入 now_pop
    if final_score >= 0:
        que.put(weight1)
    else:
        que.put(weight2)
    # chess1.show_board(time=5, title='score: '+str(final_score))


if __name__ == '__main__':
    epochs = 250
    pop_size = 2
    init_time = 3
    pop = []
    for s in range(pop_size * (2 ** init_time)):
        pop.append([[random.randint(100, 500)],
                    [random.randint(-200, 50), random.randint(-200, 50)],
                    [random.randint(-50, 200), random.randint(-100, 150), random.randint(-100, 150)],
                    [random.randint(-50, 200), random.randint(-100, 150), random.randint(-100, 150), 0],
                    random.randint(50, 300), random.randint(10, 100)])
    for s in range(init_time):
        pop = selection(pop)
    print(pop)
    start = time.time()
    for epoch in range(epochs):
        new_pop = cross_over(pop, p=0.75)
        new_pop = mutation(new_pop, p=0.25, r=50)
        new_pop = selection(new_pop)
        pop = new_pop
        print('epoch', epoch, ': ', new_pop[0])

    while len(new_pop) > 1:
        new_pop = selection(new_pop)
    print(new_pop)

    end = time.time()
    print("共用时: ", end-start)

