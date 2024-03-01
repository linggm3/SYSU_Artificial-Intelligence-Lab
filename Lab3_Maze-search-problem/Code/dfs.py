import queue
from Class import next_stage, Maze
from id_a_star import h_n


# 深度优先搜索
def depth_first_search(maze: Maze):
    sta = queue.LifoQueue()
    depth_sta = queue.LifoQueue()
    sta.put(maze.start)
    depth_sta.put(0)

    total_search_step = 0
    while not sta.empty():
        total_search_step += 1
        now = sta.get()
        now_depth = depth_sta.get()
        # 当前位置为 now
        maze.set_status(now, 'now')
        maze.show_board(0.02, now_depth)
        if maze.is_goal(now):
            maze.show_board(2, now_depth)
            break
        # 将位置为 now 的方格 设置为 已访问
        maze.set_status(now, 'visited')
        # 下一个状态
        for stage in next_stage(now):
            if maze.is_valid(stage, circle_detection=True):
                sta.put(stage)
                depth_sta.put(now_depth + 1)
    return now_depth, total_search_step


# 深度受限搜索
def depth_limit_search(maze: Maze, limit):
    # 存储位置
    sta = queue.LifoQueue()
    # 存储代价
    depth_sta = queue.LifoQueue()
    # 存储路径
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
        # 下一个状态
        for stage in next_stage(now):
            # 带环检测得不到最优解
            # if maze.is_valid(stage, circle_detection=True):

            # 带路径检测
            if maze.is_valid2(stage, now_path, path_detection=True):
                sta.put(stage)
                depth_sta.put(now_depth+1)
                stage_path = now_path.copy()
                stage_path.add(now)
                path_sta.put(stage_path)
    if maze.is_goal(now):
        return now_depth, total_search_step
    return -1, total_search_step


def id_dfs(maze: Maze, begin=1, factor=1):
    if begin == 1:
        begin = h_n(maze.start, maze.end)
    total_search_step = 0
    res = -1
    counter = begin
    while res == -1:
        res, step = depth_limit_search(maze, counter)
        total_search_step += step
        maze.reset()
        counter += factor
    return res, total_search_step
