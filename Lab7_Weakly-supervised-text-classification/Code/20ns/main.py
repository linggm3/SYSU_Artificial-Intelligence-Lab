import json

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import Dataset, DataLoader
from MyDataset import get_word2index, TextDataset, move_data
from train import TextClassifier, supervised_train, unsupervised_train


if __name__ == '__main__':
    # 定义超参数和数据路径
    vocab_size = 101316  # 词汇表大小
    embedding_dim = 200  # 词向量维度
    num_classes = 20  # 类别数
    batch_size = 1
    num_epochs = 250
    learning_rate = 0.001
    labeled_data_path = r'processed_data/labeled_dataset.csv'
    unlabeled_data_path = r'processed_data/unlabeled_dataset.csv'

    # 创建词汇表
    word2index = get_word2index(labeled_data_path, unlabeled_data_path)
    print('word_dict prepared')

    # 创建数据集实例
    labeled_dataset = TextDataset(labeled_data_path, word2index)
    unlabeled_dataset = TextDataset(unlabeled_data_path, word2index)
    print('dataset prepared')

    # 创建数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型实例
    model = TextClassifier(num_classes, vocab_size, embedding_dim).cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        if epoch == 0:
            for i in range(25):
                res = supervised_train(model, criterion, optimizer, labeled_loader)
        else:
            for i in range(1):
                res = supervised_train(model, criterion, optimizer, labeled_loader)
        rm = unsupervised_train(model, criterion, optimizer, unlabeled_loader)

        move_data(labeled_dataset, unlabeled_dataset, rm, 750)

        print(labeled_dataset.__len__())
        print(unlabeled_dataset.__len__())
        labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

        print('--------Epoch : ', epoch, '  acc : ', res, '--------')
        if unlabeled_dataset.__len__() < 10:
            torch.save(model, 'model_750_nodist.pth')
            break
