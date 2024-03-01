from copy import deepcopy
import numpy as np


class Graph:
    def __init__(self, txt_location):
        # 实现城市名和数字索引间的转换
        self.list = {}
        # 邻接表，city_a->city_b : len
        self.adj = {}
        number = 0
        # 打开相应位置的txt，只读模式,ascii编码，缩写为f
        with open(txt_location, 'r', encoding='ascii') as f:
            for item in f.readlines():
                item = item.strip('\n')  # 去除文本中的换行符
                item = item.split(sep=' ')
                # 第一行
                if len(item) < 3:
                    self.node_num = int(item[0])
                    self.edge_num = int(item[1])
                    continue
                # 新遇到的城市添加到list中记录下来
                if item[0].lower() not in self.list:
                    self.list[item[0].lower()] = number
                    self.list[number] = item[0].lower()
                    number = number + 1
                if item[1].lower() not in self.list:
                    self.list[item[1].lower()] = number
                    self.list[number] = item[1].lower()
                    number = number + 1
                # 更新邻接表
                string = item[0] + '->' + item[1]
                string = string.lower()
                self.adj[string] = int(item[2])
                string = item[1] + '->' + item[0]
                string = string.lower()
                self.adj[string] = int(item[2])
        # print(self.list)
        # print('\n')
        # print(self.adj)

    def city_name(self, a):
        a = a.lower()
        if type(a) == int:
            x = self.list[a]
        elif len(a) == 1:
            for key in self.list.keys():
                if type(key) == str and key[0] == a:
                    x = key
                    break
        else:
            x = a
        if x not in self.list:
            print("输入的城市不存在！")
        return x

    def dist(self, a, b):
        return self.adj[self.city_name(a)+'->'+self.city_name(b)]

    def dijkstra(self, a, b):
        res = open('res.txt', mode='a')
        # 起始节点
        start = self.list[self.city_name(a)]
        # 目标节点
        target = self.list[self.city_name(b)]
        print("起点为:", self.list[start], "终点为:", self.list[target], sep=' ')
        res.writelines(["start: ", self.list[start], "  end: ", self.list[target], '\n'])
        # 是否确定最短路径
        determined = []
        # 目前最短路径的距离
        distance = []
        # 最短路径的前驱节点
        pre = []
        # 初始化数组
        for i in range(0, self.node_num):
            determined.append(False)
            distance.append(0x7FFFFFFF)
            pre.append(-1)
        # 起始节点确定,距离为0
        determined[start] = True
        distance[start] = 0
        pre[start] = start
        # 更新初始距离
        for i in range(0, self.node_num):
            if self.list[start]+'->'+self.list[i] in self.adj:
                distance[i] = self.adj[self.list[start]+'->'+self.list[i]]
                pre[i] = start
        # dijkstra过程
        for e in range(1, self.node_num):
            # 找到未确定的距离最短的点
            min_dist = 0x7FFFFFFF
            min_position = -1
            for i in range(0, self.node_num):
                # 还没有确定最短路径，且目前最短路径小于min
                if not determined[i] and distance[i] < min_dist:
                    min_dist = distance[i]
                    min_position = i
            if min_position == -1:
                break
            # 将距离最短的点的最短路径确定下来
            # print(self.list[min_position] + "的最短路径已确定")
            determined[min_position] = True
            # 更新其他节点的目前最短路径
            for i in range(0, self.node_num):
                # 如果存在这样的一条道路
                if self.list[min_position]+'->'+self.list[i] in self.adj:
                    # 如果start到min_position的距离 + min_position到i的距离 < start到i的距离
                    if np.longlong(self.adj[self.list[min_position]+'->'+self.list[i]] + min_dist) < np.longlong(distance[i]):
                        # print(self.list[i] + "的最短路径长度从" + str(distance[i]) + "更新为" + str(self.adj[self.list[min_position]+'->'+self.list[i]] + min_dist))
                        # 更新距离
                        distance[i] = self.adj[self.list[min_position]+'->'+self.list[i]] + min_dist
                        # 更新前驱节点
                        pre[i] = min_position
        # print(determined)
        # print(distance)
        # print(pre)
        tmp = target
        print("最短路径为:", end=' ')
        res.write("Shortest path: ")
        print(self.list[target].title(), end='<--')
        while pre[tmp] != start:
            print(self.list[pre[tmp]].title(), end='<--')
            res.write(self.list[pre[tmp]].title() + "<--")
            tmp = pre[tmp]
        print(self.list[start].title())
        res.write(self.list[start].title() + '\n')
        print("最短路径长度为:", distance[target], sep=' ')
        res.writelines(["Length of the shorest path:", str(distance[target]), '\n\n'])
        return distance[target]


if __name__ == '__main__':
    graph = Graph('Romania.txt')
    city_a = input("请输入起始城市    ")
    city_b = input("请输入终点城市    ")
    graph.dijkstra(city_a, city_b)
