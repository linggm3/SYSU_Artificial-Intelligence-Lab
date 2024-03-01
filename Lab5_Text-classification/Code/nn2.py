import collections
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader


# word_list 缺失时返回编号0
def defaultvalue():
    return 0


class MyDataSet(Dataset):
    def __init__(self, is_train=True, wd={}):
        """
        :param is_train: 调用的是训练集(True)，还是验证集(False)
        """
        encodings = 'utf-8'
        if is_train:
            file = open('train_set.csv', mode='r', encoding=encodings)
            # 读取文本
            sentences = []
            labels = []
            labels_dict = {'sad': 0, 'disgust': 1, 'joy': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
            num_labels = []
            for line in file:
                # 分割成句子
                tmp_line = re.split(',', line)
                sentences.extend(tmp_line[0:-1])
                labels.append(tmp_line[-1].removesuffix('\n'))
                tmp_label = [0, 0, 0, 0, 0, 0]
                tmp_label[labels_dict[tmp_line[-1].removesuffix('\n')]] = 1
                num_labels.append(tmp_label)

            '''file = open('validation_set.csv', mode='r', encoding=encodings)
            for line in file:
                # 分割成句子
                tmp_line = re.split(',', line)
                sentences.extend(tmp_line[0:-1])
                labels.append(tmp_line[-1].removesuffix('\n'))
                tmp_label = [0, 0, 0, 0, 0, 0]
                tmp_label[labels_dict[tmp_line[-1].removesuffix('\n')]] = 1
                num_labels.append(tmp_label)'''

            word_list = ""
            for sentence in sentences:
                word_list += sentence + ' '
            word_list = word_list.split(' ')
            self.word_list = list(set(word_list))  # 去重
            self.word_dict = collections.defaultdict(defaultvalue)  # 从 单词 转化成 单词对应的编号（默认值为0）
            self.number_dict = collections.defaultdict(defaultvalue)  # 从 单词编号 转化成 编号对应的单词
            for i in range(len(self.word_list)):
                self.word_dict[self.word_list[i]] = i + 1
                self.number_dict[i + 1] = self.word_list[i]
            print(self.word_dict)
            print(self.number_dict)
            self.labels = num_labels
            self.sentences = sentences
            self.num_sentences = []
            for i in range(len(sentences)):
                self.num_sentences.append([])
                # print(sentences[i])
                for j in sentences[i].split(' '):
                    # print(j)
                    self.num_sentences[i].append(self.word_dict[j])
            # print(self.num_sentences)
            print(len(self.word_list))

        else:
            file = open('classification_test_set.csv', mode='r', encoding=encodings)
            # 读取文本
            sentences = []
            labels = []
            labels_dict = {'sad': 0, 'disgust': 1, 'joy': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
            num_labels = []
            for line in file:
                # 分割成句子
                tmp_line = re.split(',', line)
                sentences.extend(tmp_line[0:-1])
                labels.append(tmp_line[-1].removesuffix('\n'))
                tmp_label = [0, 0, 0, 0, 0, 0]
                tmp_label[labels_dict[tmp_line[-1].removesuffix('\n')]] = 1
                num_labels.append(tmp_label)
            self.labels = num_labels
            self.num_sentences = []
            self.sentences = sentences
            for i in range(len(sentences)):
                self.num_sentences.append([])
                # print(sentences[i])
                for j in sentences[i].split(' '):
                    # print(j)
                    self.num_sentences[i].append(wd[j])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.LongTensor(self.num_sentences[index]), self.labels[index], self.sentences[index]


# Model
class Word2Vec(nn.Module):
    def __init__(self, voc_size, out_size):
        super(Word2Vec, self).__init__()
        self.embed = nn.Embedding(voc_size, embedding_size)
        self.U = nn.Linear(embedding_size, out_size, bias=True)
        self.W = nn.Linear(embedding_size, out_size, bias=True)
        self.f = nn.Sigmoid()

    def forward(self, X):
        # X : [batch_size, voc_size]
        # print(X.size())
        out = self.embed(X)  # output_layer : [batch_size, voc_size]
        # print(out.size())
        # print(out[0][0][:].size())
        res = []
        res.append(self.f(self.U(out[0][0][:])))
        # print(res.size())
        for i in range(1, out.size()[1]):
            res.append(self.f(self.U(out[0][i][:])))
        ans = sum(res) / out.size()[1]
        return ans


if __name__ == '__main__':
    batch_size = 1
    embedding_size = 200
    voc_size = 2033
    # voc_size = 2680
    out_size = 6
    my_train_data = MyDataSet(is_train=True)
    my_test_data = MyDataSet(is_train=False, wd=my_train_data.word_dict)
    train_dataloader = DataLoader(my_train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(my_test_data, batch_size=batch_size, shuffle=True)

    model = Word2Vec(voc_size, out_size)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fun = nn.MSELoss()

    a = ['sad', 'disgust', 'joy', 'anger', 'fear', 'surprise']

    right = 0
    total = 0
    # Training
    for epoch in range(50):
        right = 0
        total = 0
        for sen, label, b in train_dataloader:
            optimizer.zero_grad()
            output = model(sen)
            # output = np.argmax(output.detach().numpy(), axis=0)
            loss = loss_fun(torch.Tensor(label), torch.Tensor(output))
            loss.backward()
            optimizer.step()
            total += 1
            if np.argmax(output.detach().numpy(), axis=0) == np.argmax(label, axis=0):
                right += 1
        print("训练集准确率: ", right / total)
        right = 0
        total = 0
        with torch.no_grad():
            for sen, label, b in validation_dataloader:
                output = model(sen)
                loss = loss_fun(torch.Tensor(label), torch.Tensor(output))
                total += 1
                if np.argmax(output.detach().numpy(), axis=0) == np.argmax(label, axis=0):
                    right += 1
        print("验证集准确率: ", right / total)
        print()



