import queue
from Class import next_stage, Maze


# 宽度优先搜索
def board_first_search(maze: Maze):
    que = queue.Queue()
    step_que = queue.Queue()
    que.put(maze.start)
    step_que.put(0)

    # 记录在队列中的元素，不让相同的元素进入队列
    in_queue_element = set()
    in_queue_element.add(maze.start)

    total_search_step = 0
    while not que.empty():
        total_search_step += 1
        now = que.get()
        in_queue_element.remove(now)

        now_step = step_que.get()
        # 当前位置为 now
        maze.set_status(now, 'now')
        maze.show_board(0.02, now_step)
        if maze.is_goal(now):
            maze.show_board(2, now_step)
            break
        # 将位置为 now 的方格 设置为 已访问
        maze.set_status(now, 'visited')
        # 下一个状态
        for stage in next_stage(now):
            if maze.is_valid(stage) and stage not in in_queue_element:
                in_queue_element.add(stage)
                que.put(stage)
                step_que.put(now_step+1)
    return now_step, total_search_step


# 双向搜索
def double_ended_search(maze: Maze):
    board = [[' ' for j in range(len(maze.maze_data[0]))] for i in range(len(maze.maze_data))]

    que = queue.Queue()
    que.put((maze.start, 's'))
    que.put((maze.end, 'e'))

    step_que = queue.Queue()
    step_que.put(0)
    step_que.put(0)

    # 记录在队列中的元素，不让相同的元素进入队列
    in_queue_element = set()
    in_queue_element.add((maze.start, 's'))
    in_queue_element.add((maze.end, 'e'))

    total_search_step = 0
    while not que.empty():
        total_search_step += 1
        now, now_state = que.get()
        now_step = step_que.get()
        in_queue_element.remove((now, now_state))

        if now_state == 's' and board[now[0]][now[1]] == 'e' or now_state == 'e' and board[now[0]][now[1]] == 's':
            break

        # 当前位置为 now
        maze.set_status(now, 'now')
        if now_state == 's':
            maze.show_board(0.02, 2 * now_step - 1)
        else:
            maze.show_board(0.02, 2 * now_step)

        # 将位置为 now 的方格 设置为 已访问
        maze.set_status(now, 'visited')
        board[now[0]][now[1]] = now_state

        # 下一个状态
        for stage in next_stage(now):
            if maze.is_valid(stage) and (stage, now_state) not in in_queue_element:
                in_queue_element.add((stage, now_state))
                que.put((stage, now_state))
                step_que.put(now_step+1)

    if now_state == 's':
        maze.show_board(2, 2 * now_step - 1)
        return 2*now_step-1, total_search_step
    else:
        maze.show_board(2, 2 * now_step)
        return 2 * now_step, total_search_step
