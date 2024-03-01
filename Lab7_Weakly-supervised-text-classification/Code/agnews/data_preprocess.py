import re
import nltk
import numpy as np
from nltk.corpus import stopwords

# nltk.download('stopwords')


def preprocess_string(text):
    # 删除标点符号和数字
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('\d+', '', text)

    # 删除 '\' 和 '\' 后面的一个字符
    text = re.sub(r'\\(.)', '', text)

    # 去除停用词
    stop_words = set(stopwords.words('english'))
    tokens = text.split()

    # 去除单个字母的 token
    tokens = [token for token in tokens if len(token) > 1]

    # 去除停用词
    tokens = [token for token in tokens if token.lower() not in stop_words]
    text = ' '.join(tokens)

    # 去除两端的空格并将多个空格合并为一个空格
    text = re.sub('\s+', ' ', text).strip()

    return text


if __name__ == '__main__':
    file_name = 'dataset.csv'
    read_file = open(r'Dataset/agnews/'+file_name, mode='r', encoding='ascii')
    keyword_file = open(r'Dataset/agnews/keywords.txt', mode='r', encoding='ascii')
    write_file_0 = open(r'processed_data/dataset.csv', mode='w', encoding='ascii')
    write_file_1 = open(r'processed_data/labeled_dataset.csv', mode='w', encoding='ascii')
    write_file_2 = open(r'processed_data/unlabeled_dataset.csv', mode='w', encoding='ascii')

    keyword = []
    for line in keyword_file:
        line = line.split(':')
        keyword.append(line[1].strip().split(','))
    print(keyword)

    avg_len = 0
    right = 0
    total = 0
    counter = 0
    for line in read_file:
        counter += 1
        res = ['']
        index = 0
        while not line[index].isdigit():
            index += 1
        while line[index].isdigit():
            res[0] = res[0] + line[index]
            index += 1
        res.append(line[index+1:])

        label = res[0]
        text = res[1]

        text = preprocess_string(text).strip()
        avg_len += len(text.split(' '))

        if len(text) != 0:
            write_file_0.write(label + '\t' + str(-1) + '\t' + text + '\n')

        break_flag = False
        class_index = []
        for word in text.split(' '):
            for i in range(len(keyword)):
                if word in keyword[i] and len(text) != 0:
                    class_index.append(i)

        if len(class_index) > 0 and len(text) != 0:
            # 众数表决
            counts = np.bincount(class_index)
            if 2 * max(counts) > sum(counts):
                # print(class_index, counts, int(np.argmax(counts)) == int(label), find_second_largest(counts))
                write_file_1.write(label + '\t' + str(int(np.argmax(counts))) + '\t' + text + '\n')
                if int(np.argmax(counts)) == int(label):
                    right += 1
                total += 1
            else:
                write_file_2.write(label + '\t' + str(-1) + '\t' + text + '\n')
        elif len(class_index) == 0 and len(text) != 0:
            write_file_2.write(label + '\t' + str(-1) + '\t' + text + '\n')

    print(right, total, right / total)
    print(avg_len / counter)
