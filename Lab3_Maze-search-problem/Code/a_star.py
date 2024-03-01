import queue
from Class import next_stage, Maze
from math import sqrt


# 采用 当前节点 到终点的 曼哈顿距离 作为 启发式函数
def h_n(now, tar):
    return abs(now[0] - tar[0]) + abs(now[1] - tar[1])


# 采用 当前节点 到终点的 欧式距离 作为 启发式函数
def h_n2(now, tar):
    return sqrt((now[0] - tar[0]) ** 2 + (now[1] - tar[1]) ** 2)


def a_star(maze: Maze, add_now_cost=True):
    # 排序函数 中 加不加 （从初始节点 到 每个已探索节点 的最小代价）
    if add_now_cost:
        add = 1
    else:
        add = 0

    # 优先队列，按启发式函数值排列，第0个元素的函数值，第1个元素是位置
    # 第2个元素是从初始节点到这个节点的最小代价，第3个元素是路径
    que = queue.PriorityQueue()

    # 启发式函数
    hn_now = h_n(maze.start, maze.end)

    # 排序函数值为 hn_now + 0，位置是 maze.start，从起点到 maze.start 代价是 0， 路径为 {}
    que.put((hn_now + 0, maze.start, 0, set([])))

    # 记录在队列中的元素，不让相同的元素进入队列
    in_queue_element = set()
    in_queue_element.add(maze.start)

    total_search_step = 0
    while not que.empty():
        total_search_step += 1
        now, now_cost, now_path = que.get()[1:4]
        in_queue_element.remove(now)
        # 当前位置为 now
        maze.set_status(now, 'now')
        maze.show_board(0.02, now_cost)
        if maze.is_goal(now):
            maze.show_board(2, now_cost)
            break
        # 将位置为 now 的方格 设置为 已访问
        maze.set_status(now, 'visited')
        # 下一个状态
        for stage in next_stage(now):
            # 带环检测可以得到最优解（h_n单调）
            if maze.is_valid(stage, circle_detection=True) and stage not in in_queue_element:
                # 在路径中添加pos
                hn_stage = h_n(stage, maze.end)
                stage_path = now_path.copy()
                stage_path.add(now)
                que.put((hn_stage+now_cost+add, stage, now_cost+add, stage_path))
                in_queue_element.add(stage)

    return now_cost, total_search_step

