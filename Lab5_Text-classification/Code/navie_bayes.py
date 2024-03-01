import re
import math


class NaiveBayes:
    def __init__(self):
        self.vocabulary = set()    # 词汇表
        self.word_count = {}       # 每个类别中每个单词出现的次数
        self.category_count = {}   # 每个类别出现的次数

    # 将文本数据转换为单词列表
    def text_parse(self, text):
        words = re.split(r'\W+', text)
        return [word.lower() for word in words if len(word) > 2]

    # 计算每个类别中每个单词出现的次数
    def fit(self, dataset, labels):
        for i in range(len(dataset)):
            text = dataset[i]
            label = labels[i]
            # 统计类别出现的次数（未见过的类别次数置0）
            self.category_count[label] = self.category_count.get(label, 0) + 1
            # 将文本数据转换为单词列表
            words = self.text_parse(text)
            for word in words:
                # 将单词添加到词汇表中
                self.vocabulary.add(word)
                # 统计每个类别中每个单词出现的次数
                if label not in self.word_count:
                    self.word_count[label] = {}
                self.word_count[label][word] = self.word_count[label].get(word, 0) + 1

    # 计算单词在类别中出现的概率
    def word_prob(self, word, category):
        # 计算类别中单词出现的总次数
        category_word_count = sum(self.word_count[category].values())
        # 计算单词在类别中出现的次数
        word_count = self.word_count[category].get(word, 0)
        # 计算概率，并使用拉普拉斯平滑避免出现概率为0的情况
        return (word_count + 2) / (category_word_count + 2*len(self.vocabulary))

    # 计算文本数据属于每个类别的概率
    def predict_prob(self, text):
        # 将文本数据转换为单词列表
        words = self.text_parse(text)
        # 初始化概率为1，避免出现概率为0的情况
        prob = {category: 1 for category in self.category_count.keys()}
        for category in self.category_count.keys():
            for word in words:
                # 计算单词在类别中出现的概率，并累乘
                prob[category] *= self.word_prob(word, category)
            # 计算文本数据属于该类别的概率（归一化）
            prob[category] *= self.category_count[category] / sum(self.category_count.values())
        return prob

    # 预测文本数据的类别
    def predict(self, text):
        prob = self.predict_prob(text)
        # 返回概率最大的类别
        return max(prob, key=prob.get)


if __name__ == '__main__':
    # 创建朴素贝叶斯分类器
    nb = NaiveBayes()

    file = open('train_set.csv', mode='r', encoding='ascii')
    # 读取文本
    train_dataset = []
    train_labels = []
    for line in file:
        # 分割成句子
        tmp_line = re.split(',', line)
        train_dataset.extend(tmp_line[0:-1])
        train_labels.append(tmp_line[-1].removesuffix('\n'))
    print(train_dataset[0:5])
    print(train_labels[0:5])

    # 训练
    nb.fit(train_dataset, train_labels)

    # 预测新数据
    # file = open('train_set.csv', mode='r', encoding='ascii')
    file = open('classification_test_set.csv', mode='r', encoding='ascii')
    file2 = open('21307077_nb_classification.csv', mode='w', encoding='ascii')
    # file = open('validation_set.csv', mode='r', encoding='ascii')
    # 读取文本
    test_dataset = []
    test_labels = []
    right = 0
    total = 0
    for line in file:
        # 分割成句子
        tmp_line = re.split(',', line)
        test_dataset.extend(tmp_line[0:-1])
        test_labels.append(tmp_line[-1].removesuffix('\n'))
        print(nb.predict(test_dataset[-1]), test_labels[-1])  # 输出：positive
        file2.write(test_dataset[-1] + ',' + nb.predict(test_dataset[-1]) + '\n')
        if nb.predict(test_dataset[-1]) == test_labels[-1]:
            right += 1
        total += 1
    print(right / total)
