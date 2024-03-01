import re
import string


def remove_punctuation(text_):
    # 删除标点符号
    text_ = re.sub(r'[^\w\s]', '', text_)
    # 删除数字
    text_ = re.sub(r'\d+', '', text_)
    # 删除冠词
    text_ = re.sub(r'(?<=[^\w\s])(?=[\w\s])', '', text_)
    # 删除'\'和'\'后面的一个字符
    text_ = text_.replace('\\', '')
    # 去除两端的空格
    text_ = text_.lstrip('\t \n')
    # 将多个空格合并为一个空格后返回
    text_ = ' '.join(text_.split())
    return text_


if __name__ == '__main__':
    file_name = 'valid.txt'
    read_file = open(r'20ns/'+file_name, mode='r', encoding='ascii')
    write_file = open(r'processed_data/'+file_name, mode='w', encoding='ascii')
    for line in read_file:
        line = line.split('\t')
        label = line[0]
        text = line[1]
        write_file.write(label + '\t' + remove_punctuation(text) + '\n')
