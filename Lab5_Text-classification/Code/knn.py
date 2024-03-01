import math
import re


# 将文本转换为向量表示
def text2vec(text):
    words = text.split()
    vec = {}
    for word in words:
        vec[word] = vec.get(word, 0) + 1
    return vec


# 计算两个向量之间的余弦相似度
def cosSimilarity(vec1, vec2):
    dotProduct = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for key in vec1:
        # key in vec2 表示对应元素相乘，如果不对应则一方为 0，忽略
        if key in vec2:
            # Sigma(xi * yi)
            dotProduct += vec1[key] * vec2[key]
        norm1 += vec1[key] ** 2  # Sigma(xi^2)
    for key in vec2:
        norm2 += vec2[key] ** 2  # Sigma(yi^2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    else:
        # Sigma(xi*yi) / (sqrt(Sigma(xi^2) + sqrt(Sigma(yi^2) )
        return dotProduct / (math.sqrt(norm1) * math.sqrt(norm2))


# 读取数据集
def loadDataset(filename):
    dataset = []
    labels = []
    with open(filename) as file:
        for line in file:
            line = line.strip().split(',')
            dataset.append(text2vec(line[0]))
            labels.append(line[1])
    return dataset, labels


# k-近邻算法
def knnClassify(inputVec, dataset, labels, k):
    similarities = []
    for i in range(len(dataset)):
        sim = cosSimilarity(inputVec, dataset[i])
        similarities.append((sim, labels[i]))
    similarities.sort(reverse=True)
    classCount = {}
    for i in range(k):
        voteLabel = similarities[i][1]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]


# 加载数据集
dataset, labels = loadDataset('train_set.csv')


# 设置k值
k = 4
file = open('classification_test_set.csv', mode='r', encoding='ascii')
file2 = open('21307077_knn_classification.csv', mode='w', encoding='ascii')
# file = open('21881234_pinyin_model_classification.csv', mode='r', encoding='ascii')
# 读取文本
right = 0
total = 0
for line in file:
    # 分割成句子
    tmp_line = re.split(',', line)
    test_data = tmp_line[0:-1][0]
    test_label = tmp_line[-1].removesuffix('\n')
    testVec = text2vec(test_data)
    print(knnClassify(testVec, dataset, labels, k), test_label)
    file2.write(test_data + ',' + knnClassify(testVec, dataset, labels, k) + '\n')
    if knnClassify(testVec, dataset, labels, k) == test_label:
        right += 1
    total += 1
print(right / total)