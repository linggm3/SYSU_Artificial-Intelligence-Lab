from matplotlib import pyplot as plt
import numpy as np
import queue


def next_stage(pos):
    return (pos[0] + 1, pos[1]), (pos[0], pos[1] + 1), (pos[0] - 1, pos[1]), (pos[0], pos[1] - 1)


class Maze:
    def __init__(self, data_path):
        # 打开 MazeData 文件
        file = open(data_path, mode='r', encoding='ascii')

        # 颜色，分别是 通路，墙，已走过的路，当前位置，目标位置
        self.color = [[255, 255, 204], [128, 128, 128], [135, 206, 235], [0, 0, 255], [255, 0, 0]]
        self.maze_data = []

        # 读取 MazeData 文件
        for line in file:
            tmp = (list(line.strip('\n')))
            self.maze_data.append(tmp)

        # 画图展示 迷宫状态
        self.maze_board = np.zeros((len(self.maze_data), len(self.maze_data[0]), 3), dtype=np.uint8)
        for i in range(len(self.maze_data)):
            for j in range(len(self.maze_data[0])):
                # 如果是 起点
                if self.maze_data[i][j] == 'S':
                    self.start = (i, j)
                    self.maze_board[i][j] = self.color[3]
                    continue
                # 如果是 终点
                elif self.maze_data[i][j] == 'E':
                    self.end = (i, j)
                    self.maze_board[i][j] = self.color[4]
                    continue
                self.maze_data[i][j] = int(self.maze_data[i][j])
                # 如果是 通路
                if self.maze_data[i][j] == 0:  # 可行通路
                    self.maze_board[i][j] = self.color[0]
                # 如果是 墙
                elif self.maze_data[i][j] == 1:  # 墙
                    self.maze_board[i][j] = self.color[1]
        self.show_board(1)

    # 重置状态
    def reset(self):
        for i in range(len(self.maze_data)):
            for j in range(len(self.maze_data[0])):
                if self.maze_data[i][j] == 2 or self.maze_data[i][j] == 3:
                    self.maze_data[i][j] = 0
                    self.maze_board[i][j] = self.color[0]
        self.maze_board[self.start[0]][self.start[1]] = self.color[3]
        self.maze_board[self.end[0]][self.end[1]] = self.color[4]

    def is_goal(self, now):
        return now[0] == self.end[0] and now[1] == self.end[1]

    # 移动到这个位置是否合法（可设置环检测）
    def is_valid(self, pos, circle_detection=True):
        if 0 < pos[0] < len(self.maze_data) and 0 < pos[1] < len(self.maze_data[0]):
            # 不能是墙。如果带环检测，就不能是访问过的。
            if self.maze_data[pos[0]][pos[1]] != 1 and ((not circle_detection) or self.maze_data[pos[0]][pos[1]] != 2):
                return True
        return False

    # 移动到这个位置是否合法（可设置路径检测）
    def is_valid2(self, pos, path, path_detection=True):
        if 0 < pos[0] < len(self.maze_data) and 0 < pos[1] < len(self.maze_data[0]):
            # 不能是墙。
            if self.maze_data[pos[0]][pos[1]] != 1 and  \
                    ((not path_detection) or pos not in path):
                return True
        return False

    # 设置状态（已访问，未访问，当前状态）
    def set_status(self, pos, status):
        if status == 'now':
            self.maze_data[pos[0]][pos[1]] = 3
            self.maze_board[pos[0]][pos[1]] = self.color[3]
        elif status == 'visited':
            self.maze_data[pos[0]][pos[1]] = 2
            self.maze_board[pos[0]][pos[1]] = self.color[2]
        elif status == 'unvisited':
            self.maze_data[pos[0]][pos[1]] = 0
            self.maze_board[pos[0]][pos[1]] = self.color[0]

    # 展示迷宫
    def show_board(self, time, step=0):
        # 如果不想查看算法的可视化过程，请把整段注释掉
        plt.ion()
        fig = plt.figure('frame')
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.axis('off')
        ax1.imshow(self.maze_board, interpolation='nearest')
        plt.title('step: ' + str(step), fontdict={'fontname': 'Arial', 'fontsize': 16})
        ax1.axis('off')
        plt.pause(time)
        fig.clf()
