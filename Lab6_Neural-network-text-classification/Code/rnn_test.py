import collections
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader


# word2vec 缺失时返回默认值
def default_value():
    return [0 for j in range(50)]


label_dict = {}


class MyDataSet(Dataset):
    def __init__(self, pwd):
        # 预训练词向量
        token_file = open("glove.6B.50d.txt", mode='r', encoding='utf8')
        # 数据集，格式为 label \t text
        file = open(pwd, mode='r', encoding='ascii')
        # 词典，分词 : 50维词向量
        self.word2vec = collections.defaultdict(default_value)
        # 利用预训练词向量文件，初始化词典
        for line in token_file:
            data = line.split(' ')
            if len(data) < 51:
                continue
            # 类型转化，从 str 转为 LongTensor
            for i in range(1, 51):
                data[i] = float(data[i])
            self.word2vec[data[0]] = data[1:]
        token_file.close()

        self.str_data = []  # 字符串形式的 data
        self.str_labels = []  # 字符串形式的 label
        self.vec_data = []  # 向量形式的 data
        self.vec_labels = []  # 向量形式的 label
        self.num_labels = []  # 数字形式的 label
        for line in file:
            label_, text_ = line.split('\t')
            text_ = text_.split(' ')  # 以空格为分隔，分割 token
            self.str_data.append(text_)  # 新增 字符串形式的 data
            self.str_labels.append(label_)  # 新增 字符串形式的 label
            self.vec_data.append([])  # 新增 向量形式的 data
            self.vec_labels.append([0 for i in range(len(label_dict))])  # 新增 向量形式的 label
            self.vec_labels[-1][label_dict[label_]] = 1
            self.num_labels.append(label_dict[label_])
            for word_ in text_:  # 对于 text 里的每个分词
                self.vec_data[-1].append(self.word2vec[word_])  # 每个 token 一个 []
        file.close()

    def __len__(self):
        return len(self.str_data)

    def __getitem__(self, index):
        return torch.Tensor(self.vec_data[index]), self.num_labels[index], self.str_data[index]


class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, bidirectional=True):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # x 的形状: (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out 的形状: (batch_size, seq_length, hidden_dim)
        # 仅使用最后一个时间步的输出
        final_output = lstm_out[:, -1, :]
        final_output = self.dropout(final_output)
        # final_output 的形状: (batch_size, hidden_dim)
        out = self.fc(final_output)
        # out 的形状: (batch_size, num_classes)
        return out


if __name__ == '__main__':
    # 读取 labels.txt，初始化 字符串形式的 label 的编号
    label_file = open(r"processed_data/labels.txt", mode='r', encoding='ascii')
    counter = 0
    num_dict = {}
    for item in label_file:
        label_dict[item.strip()] = counter
        num_dict[counter] = item.strip()
        counter += 1
    print(label_dict)

    batch_size = 1
    model = torch.load('../Result/rnn_epoch40')
    my_test_data = MyDataSet(r'processed_data/test.txt')  # 测试
    test_dataloader = DataLoader(my_test_data, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():  # 迁移到 GPU 上训练
        model = model.cuda()

    # file = open('../Result/21307077_lingguoming_rnn_classification.csv', mode='w', encoding='ascii')

    model.eval()
    with torch.no_grad():  # 不需要计算梯度
        right = 0
        total = 0
        for sen, label, str_sen in test_dataloader:
            if torch.cuda.is_available():
                sen = sen.cuda()
                label = label.cuda()
            output = model(sen)
            total += 1
            '''for i in range(len(str_sen)):
                str_sen[i] = str_sen[i][0]
            file.write(str(num_dict[np.argmax(output.cpu().detach().numpy(), axis=1)[0]]) + '\t' + ' '.join(str_sen))'''
            if np.argmax(output.cpu().detach().numpy(), axis=1) == label.cpu().detach().numpy():
                right += 1
        print("测试集准确率: ", right / total)


