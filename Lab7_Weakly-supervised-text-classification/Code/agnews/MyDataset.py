import json
from math import sqrt
import torch
import numpy as np
from torch.utils.data import Dataset
vocab_size = 37651
class_num = 4


def get_word2index(labeled_data_path_, unlabeled_data_path_, first=False):
    # 构建词汇表
    if first:
        word2index = {}
        index = 1  # 词汇表索引从1开始，0用于表示未知词汇
        with open(labeled_data_path_, 'r', encoding='utf-8') as file:
            for line in file:
                _, __, text = line.strip().split('\t')
                for word in text.split():
                    if word not in word2index:
                        word2index[word] = index
                        index += 1
        with open(unlabeled_data_path_, 'r', encoding='utf-8') as file:
            for line in file:
                _, __, text = line.strip().split('\t')
                for word in text.split():
                    if word not in word2index:
                        word2index[word] = index
                        index += 1
        print(index)
        tf = open("word_dict_agnews.json", "w")
        json.dump(word2index, tf)
        tf.close()
    else:
        tf = open("word_dict_agnews.json", "r")
        word2index = json.load(tf)
        tf.close()
    return word2index


def transform_list(input_list, length):
    output_list = [0] * length  # 创建一个长度为指定长度的初始列表，全部填充为0
    for i in range(len(input_list)):
        if input_list[i] < length:
            output_list[input_list[i]] += 1
    return output_list


def vec_add(v1, v2):
    if len(v1) != len(v2):
        print('vec_add failed')
        return
    ans = [0 for _ in range(len(v1))]
    for i in range(len(v1)):
        ans[i] = v1[i] + v2[i]
    return ans


def dist_cal(v1, v2):
    if len(v1) != len(v2):
        print('dist_cal failed')
        return
    ans = 0
    for i in range(len(v1)):
        ans += (v1[i] - v2[i]) ** 2
    ans = sqrt(ans)
    return ans


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, data_path_, word2index_):
        self.path = data_path_
        self.data = []
        self.real_labels = []  # 实际标签
        self.known_labels = []  # 知道的标签
        self.count = []  # 每个类别的数量（伪标签）
        self.class_center = []  # 每个类别的中心（伪标签）
        self.word2index = word2index_
        self.load_data()

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as file:
            for line in file:
                real_label, known_label, text = line.strip().split('\t')
                text_indices = [self.word2index.get(word, 0) for word in text.split()]
                self.data.append(text_indices)
                self.real_labels.append(int(real_label))
                self.known_labels.append(int(known_label))

            '''# 计算每个类别的数量，中心（伪标签）
            self.count = [0 for j in range(class_num)]  # 每个类别的数量（伪标签）
            self.class_center = [[0 for l in range(vocab_size)] for j in range(class_num)]  # 每个类别的数量（伪标签）
            # 对每个数据
            for i in range(len(self.data)):
                self.count[int(self.known_labels[i])] += 1
                # 将 词向量 转化为 词袋表示
                tmp_data = transform_list(self.data[int(self.known_labels[i])], vocab_size)
                self.class_center[int(self.known_labels[i])] = vec_add(self.class_center[int(self.known_labels[i])], tmp_data)
            # 平均，计算中心
            for i in range(class_num):
                for j in range(len(self.class_center[i])):
                    if self.count[i] != 0:
                        self.class_center[i][j] /= self.count[i]'''

    def __getitem__(self, index):
        return torch.LongTensor(self.data[index]), self.real_labels[index], self.known_labels[index]

    def __len__(self):
        return len(self.data)

    def add(self, text_, known_label_, real_label_):
        known_label_ = int(known_label_)
        real_label_ = int(real_label_)
        self.data.append(text_)
        self.real_labels.append(real_label_)
        self.known_labels.append(known_label_)

        '''# 重新计算类别中心
        tmp_data = transform_list(text_, vocab_size)
        for j in range(len(self.class_center[known_label_])):
            self.class_center[known_label_][j] *= self.count[known_label_]
        self.class_center[known_label_] = vec_add(self.class_center[known_label_], tmp_data)
        self.count[known_label_] += 1
        for j in range(len(self.class_center[known_label_])):
            self.class_center[known_label_][j] /= self.count[known_label_]'''

    def remove(self, text_):
        text_ = list(text_.numpy())
        for idx, data in enumerate(self.data):
            if data == text_:
                self.data.pop(idx)
                self.real_labels.pop(idx)
                self.known_labels.pop(idx)
                break

    def update(self, data_path):
        self.data.clear()
        self.real_labels.clear()
        self.known_labels.clear()
        self.load_data()


def move_data(labeled_dataset, unlabeled_dataset, rm, num=500):
    '''for i in range(len(rm)):
        tmp_data = transform_list(rm[i][1][0], vocab_size)
        dist = [0 for _ in range(len(labeled_dataset.class_center))]
        for j in range(len(dist)):
            dist[j] = dist_cal(tmp_data, labeled_dataset.class_center[j])
        factor = sum(dist)
        for j in range(len(dist)):
            dist[j] /= factor
        # (置信度 - 0.9) * 10
        rm[i][0] = (rm[i][0] - 0.99) * 1000
        rm[i][0] += (1 / class_num) / dist[rm[i][1][2]]
        if i % 200 == 0:
            print((1 / class_num) / dist[rm[i][1][2]], rm[i][0] - (1 / class_num) / dist[rm[i][1][2]],
                    (1 / class_num))'''

    rm.sort(key=lambda x: x[0], reverse=True)
    # print(rm)
    counter = 0
    for item in rm:
        labeled_dataset.add(item[1][0], item[1][1], item[1][2])
        unlabeled_dataset.remove(item[1][0])
        if counter > num:
            return
        counter += 1
