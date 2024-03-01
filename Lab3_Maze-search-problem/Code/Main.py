from bfs import board_first_search, double_ended_search
from dfs import depth_first_search, depth_limit_search, id_dfs
from id_a_star import id_a_star
from a_star import a_star
from Class import Maze
from time import sleep
import timeit


if __name__ == '__main__':
    print('\n如果不想观看可视化过程，请把Class.py的show_board函数的整段注释掉\n')
    maze = Maze('MazeData1.txt')
    run_time_data = [[0, 0, 0] for j in range(6)]
    for j in range(6):
        start = timeit.default_timer()
        if j == 0:
            print('宽度优先搜索', end=' ')
            run_time_data[j][0:2] = board_first_search(maze)
        elif j == 1:
            print('双向搜索   ', end=' ')
            run_time_data[j][0:2] = double_ended_search(maze)
        elif j == 2:
            print('深度优先搜索', end=' ')
            run_time_data[j][0:2] = depth_first_search(maze)
        elif j == 3:
            print('迭代加深搜索', end=' ')
            run_time_data[j][0:2] = id_dfs(maze)
        elif j == 4:
            print('A*搜索    ', end=' ')
            run_time_data[j][0:2] = a_star(maze, add_now_cost=True)
        elif j == 5:
            print('IDA*搜索  ', end=' ')
            run_time_data[j][0:2] = id_a_star(maze, add_now_cost=True)
        end = timeit.default_timer()
        run_time_data[j][2] = str(end - start)
        print(' 搜索结果:', run_time_data[j][0], end='  ', sep='  ')
        print('搜索格数:', run_time_data[j][1], end='  ', sep='  ')
        print('运行时间:', end - start)
        maze.reset()

print('\n如果想查看算法的可视化过程，请把Class.py的show_board函数的注释去掉')