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
    def __init__(self, pwd, num=80):
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
            if len(self.vec_data[-1]) > num:
                self.vec_data[-1] = self.vec_data[-1][0:num]
            elif len(self.vec_data[-1]) < num:
                while len(self.vec_data[-1]) < num:
                    self.vec_data[-1].append([0 for i in range(50)])
            if len(self.vec_data[-1]) != num:
                print('ERROR')
        file.close()

    def __len__(self):
        return len(self.num_labels)

    def __getitem__(self, index):
        return torch.Tensor(self.vec_data[index]), self.num_labels[index], self.str_data[index]


class Model(nn.Module):
    def __init__(self, word_vec_size, out_size, num=80):
        super(Model, self).__init__()
        self.word_vec_size = word_vec_size
        self.out_size = out_size
        self.num = num
        self.cov = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 50), stride=1, padding=0, bias=True),
            # nn.MaxPool2d(kernel_size=(2, 1), stride=2),
            nn.Tanh(),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(num-2)*8, out_features=(num-2)*2),
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=(num-2)*2, out_features=20),
        )
        self.softmax = nn.Softmax()

    def forward(self, X):
        X = self.cov(X)
        # print(X.size())
        X = X.view(-1, (self.num - 2) * 8)
        # print(X.size())
        X = self.linear(X)
        # print(X.size())
        return X


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
    model = torch.load('../Result/cnn_num80_epoch50')  # 实例化模型
    my_test_data = MyDataSet(r'processed_data/test.txt')  # 验证集
    test_dataloader = DataLoader(my_test_data, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():  # 迁移到 GPU 上训练
        model = model.cuda()

    # file = open('../Result21307077_lingguoming_CNN_classification.csv', mode='w', encoding='ascii')

    right = 0
    total = 0
    model.eval()
    with torch.no_grad():  # 不需要计算梯度
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
