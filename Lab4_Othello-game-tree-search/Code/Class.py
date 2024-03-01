from matplotlib import pyplot as plt
import numpy as np
import multiprocessing


class Chess:
    def __init__(self, w):
        self.chess_board = np.zeros((8, 8, 3), dtype=np.uint8)
        self.board = [[0 for j in range(8)] for i in range(8)]
        for i in range(8):
            for j in range(8):
                # 棋盘颜色
                self.chess_board[i][j][0] = 210
                self.chess_board[i][j][1] = 190
                self.chess_board[i][j][2] = 150
        # 初始棋子
        self.board[3][3] = -1
        self.board[3][4] = 1
        self.board[4][3] = 1
        self.board[4][4] = -1
        # 判断落子合法性的检索方向
        self.dr = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.dc = [-1, 0, 1, -1, 1, -1, 0, 1]
        # 各点的权值表
        self.wmap = np.array([[500, -25, 10, 5, 5, 10, -25, 500],
                            [-25, -45, 1, 1, 1, 1, -45, -25],
                            [10, 1, 3, 2, 2, 3, 1, 10],
                            [5, 1, 2, 1, 1, 2, 1, 5],
                            [5, 1, 2, 1, 1, 2, 1, 5],
                            [10, 1, 3, 2, 2, 3, 1, 10],
                            [-25, -45, 1, 1, 1, 1, -45, -25],
                            [500, -25, 10, 5, 5, 10, -25, 500]])
        self.weight = w
        self.weight2 = w[4]
        self.weight3 = w[5]
        self.wmap = np.array([[w[0][0], w[1][0], w[2][0], w[3][0], w[3][0], w[2][0], w[1][0], w[0][0]],
                              [w[1][0], w[1][1], w[2][1], w[3][1], w[3][1], w[2][1], w[1][1], w[1][0]],
                              [w[2][0], w[2][1], w[2][2], w[3][2], w[3][2], w[2][2], w[2][1], w[2][0]],
                              [w[3][0], w[3][1], w[3][2], w[3][3], w[3][3], w[3][2], w[3][1], w[3][0]],
                              [w[3][0], w[3][1], w[3][2], w[3][3], w[3][3], w[3][2], w[3][1], w[3][0]],
                              [w[2][0], w[2][1], w[2][2], w[3][2], w[3][2], w[2][2], w[2][1], w[2][0]],
                              [w[1][0], w[1][1], w[2][1], w[3][1], w[3][1], w[2][1], w[1][1], w[1][0]],
                              [w[0][0], w[1][0], w[2][0], w[3][0], w[3][0], w[2][0], w[1][0], w[0][0]]])
        self.last_drop = [-10, -10]

    # 评价函数（第一版只有子数，第二版增加了角位和行动力）
    def score(self, player):
        now_score = 0
        # 棋盘 与 权值矩阵 点乘
        score_board = np.multiply(self.board, self.wmap)
        # 行动力
        for i in range(8):
            for j in range(8):
                # 各位置权值
                now_score += score_board[i][j]
                # 子数
                now_score += self.board[i][j] * self.weight3
                # 行动力
                if self.is_valid(i, j, 1):
                    now_score += self.weight2
                if self.is_valid(i, j, -1):
                    now_score -= self.weight2
        return now_score

    # 最终得分
    def final_score(self):
        score = 0
        for i in range(8):
            for j in range(8):
                score += self.board[i][j]
        return score * 10000

    # 判断落子合法性
    def is_valid(self, row, col, player):
        if self.board[row][col] != 0:
            return False, []
        # 各个方向是否可以翻转
        flag = False
        valid_direction = []
        for i in range(8):
            if 0 <= row + self.dr[i] < 8 and 0 <= col + self.dc[i] < 8:
                r = row + self.dr[i]
                c = col + self.dc[i]
                # 第一次调用时，确保位置是对方棋子
                if self.board[r][c] == -player:
                    # 如果此方向可以翻转，则置 flag 为 True，valid_direction 加上 i
                    if self.valid_search(r, c, i, player):
                        flag = True
                        valid_direction.append(i)
        return flag, valid_direction

    # 判断落子合法性，注意第一次调用时，确保当前位置是对方棋子
    def valid_search(self, row, col, direction, player):
        # 判断下个位置是否越界
        if 0 <= row + self.dr[direction] < 8 and 0 <= col + self.dc[direction] < 8:
            # 下个位置
            row = row + self.dr[direction]
            col = col + self.dc[direction]
            # 判断下个位置是否是对方棋子，如果是，判断下下个位置
            if self.board[row][col] == -player:
                return self.valid_search(row, col, direction, player)
        # 如果下个位置是己方棋子，则可行
        if self.board[row][col] == player:
            return True
        # 否则落子不合法
        return False

    def has_valid(self, player):
        for i in range(8):
            for j in range(8):
                if self.is_valid(i, j, player):
                    return True
        return False

    # 翻转棋子，要求位置合法
    def turn_over(self, row, col, valid_direction, player):
        # 在各个翻转方向上翻转的棋子数量
        steps = []
        for direction in valid_direction:
            r = row
            c = col
            counter = 0
            while 0 <= r + self.dr[direction] < 8 and 0 <= c + self.dc[direction] < 8:
                r += self.dr[direction]
                c += self.dc[direction]
                if self.board[r][c] == -player:
                    self.board[r][c] = player
                    counter += 1
                elif self.board[r][c] == player:
                    steps.append(counter)
                    break
                else:
                    print('ERROR')
        return valid_direction, steps

    # dfs 回溯要用到，撤回翻转操作
    def withdraw(self, row, col, directions, steps):
        self.board[row][col] = 0
        dire_counter = 0
        # 对每个方向
        for i in directions:
            for j in range(1, steps[dire_counter]+1):
                r = row + self.dr[i] * j
                c = col + self.dc[i] * j
                self.board[r][c] = -self.board[r][c]
            dire_counter += 1

    def drops(self, row, col, player):
        valid_or_not, valid_direction = self.is_valid(row, col, player)
        if valid_or_not:
            # 如果位置合法，则下子
            self.board[row][col] = player
            # 如果位置合法，则下子后翻转棋子
            [directions, steps] = self.turn_over(row, col, valid_direction, player)
            self.last_drop = [row, col]
            return True, directions, steps
        else:
            return False, [], []

    def minimax_search_with_abcut(self, ceng, max_ceng, player, show=False, max_alpha=float('-inf'), min_beta=float('inf')):
        # ceng 为偶数时 Max 玩家下子
        if ceng == 60:  # 棋局结束
            return self.final_score(), []
        if ceng == max_ceng:  # 搜到最大层
            return self.score(player), []
        pos = []  # 解的位置
        res = []  # 解的分数
        counter = 0
        for i in range(8):  # 行
            for j in range(8):  # 列
                if self.is_valid(i, j, player)[0]:  # 位置有效
                    [directions, steps] = self.drops(i, j, player)[1:3]  # 下子
                    res.append(self.minimax_search_with_abcut(ceng + 1, max_ceng, -player, show, max_alpha, min_beta)[0])
                    pos.append([i, j])
                    self.withdraw(i, j, directions=directions, steps=steps)  # 撤回，回溯
                    if player == 1:  # Max玩家
                        if res[counter] >= min_beta:  # Max玩家分数 大于 祖先min节点 的 最小beta
                            break  # 剪枝
                        elif res[counter] > max_alpha:  # 维护 最大alpha
                            max_alpha = res[counter]
                    elif player == -1:  # Min玩家
                        if res[counter] <= max_alpha:  # Min玩家分数 小于 祖先max节点 的 最大alpha
                            break  # 剪枝
                        elif res[counter] < min_beta:  # 维护 最小beta
                            min_beta = res[counter]
                    counter += 1
                    # print(i, j, directions, steps)
                    if show:
                        self.show_board(title='withdraw')
        # 没有位置可走
        if len(res) == 0:
            return self.minimax_search_with_abcut(ceng+1, max_ceng, -player, show)[0], []
        if player == 1:  # Max玩家
            return max(res), pos[res.index(max(res))]
        else:  # Min玩家
            return min(res), pos[res.index(min(res))]

    def show_board(self, time=0.6, title=''):
        plt.ion()
        fig = plt.figure('frame')
        ax1 = fig.add_subplot(1, 1, 1)
        #ax1.axis('off')
        ax1.imshow(self.chess_board, interpolation='nearest')
        plt.title(title, fontdict={'fontname': 'Arial', 'fontsize': 16})

        for i in range(7):
            x = [0.5 + i, 0.5 + i]
            y = [-0.5, 7.5]
            plt.plot(x, y, color='#888', linewidth=3)
            plt.plot(y, x, color='#888', linewidth=3)

        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 1:
                    circle = plt.Circle((j, i), 0.3, color='black')
                    ax1.add_artist(circle)
                elif self.board[i][j] == -1:
                    circle = plt.Circle((j, i), 0.3, color='w')
                    ax1.add_artist(circle)
        circle2 = plt.Circle((self.last_drop[1], self.last_drop[0]), 0.1, color='r')

        ax1.add_artist(circle2)
        plt.pause(time)
        fig.clf()
        pass
