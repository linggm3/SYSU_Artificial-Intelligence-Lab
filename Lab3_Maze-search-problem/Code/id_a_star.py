import queue
from Class import next_stage, Maze


# 采用 当前节点 到终点的 曼哈顿距离 作为 启发式函数
def h_n(now, tar):
    return abs(now[0] - tar[0]) + abs(now[1] - tar[1])


def id_a_star_limit(maze: Maze, limit, add_now_cost=True):
    # 排序函数 中 加不加 （从初始节点 到 每个已探索节点 的最小代价）
    if add_now_cost:
        add = 1
    else:
        add = 0
    # 记录位置
    sta = queue.LifoQueue()
    # 记录代价
    depth_sta = queue.LifoQueue()
    # 记录路径
    path_sta = queue.LifoQueue()

    sta.put(maze.start)
    depth_sta.put(0)
    path_sta.put(set())

    total_search_step = 0
    while not sta.empty():
        total_search_step += 1
        now = sta.get()
        now_depth = depth_sta.get()
        now_path = path_sta.get()

        if now_depth > limit:
            continue

        # 当前位置为 now
        maze.set_status(now, 'now')
        maze.show_board(0.02, now_depth)
        if maze.is_goal(now):
            maze.show_board(2, now_depth)
            break
        # 将位置为 now 的方格 设置为 已访问
        maze.set_status(now, 'visited')

        # 下个状态 的 启发式函数值+已知代价
        hn_plus_gn = [0, 0, 0, 0]
        stages = next_stage(now)
        # 下一个状态 优先扩展 启发式函数值+已知代价 最小的 状态
        for i in range(len(stages)):
            # 带环检测得不到最优解
            # if maze.is_valid(stages[i], circle_detection=True):

            # 带路径检测
            if maze.is_valid2(stages[i], now_path, path_detection=True):
                # 下个状态 的 启发式函数值+已知代价
                hn_plus_gn[i] = h_n(stages[i], maze.end) + now_depth + 1

        while max(hn_plus_gn) > 0:
            max_pos = hn_plus_gn.index(max(hn_plus_gn))
            # 在路径中添加pos
            hn_stage = h_n(stages[max_pos], maze.end)
            stage_path = now_path.copy()
            stage_path.add(now)
            sta.put(stages[max_pos])
            depth_sta.put(now_depth + 1)
            path_sta.put(stage_path)
            hn_plus_gn[max_pos] = 0

    if maze.is_goal(now):
        return now_depth, total_search_step
    return -1, total_search_step


def id_a_star(maze: Maze, begin=1, factor=1, add_now_cost=True):
    if begin == 1:
        begin = h_n(maze.start, maze.end)
    total_search_step = 0
    res = -1
    counter = begin
    while res == -1:
        res, step = id_a_star_limit(maze, counter, add_now_cost=True)
        total_search_step += step
        maze.reset()
        counter += factor
    return res, total_search_step
