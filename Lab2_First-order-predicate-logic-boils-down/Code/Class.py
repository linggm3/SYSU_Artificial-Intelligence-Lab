# 谓词类，存储谓词以及谓词里的各个公式
class WeiCi:
    def __init__(self, input_str):
        self.data = []
        # 非空
        if input_str:
            # 参考input_clause函数，输入str可能第一个元素是','
            if input_str[0] == ',':
                # 去除 ','
                input_str = input_str[1:]
            tmp = ""
            # 以 '(' ',' ')' 将谓词公式分解，data第一项是谓词，后面是谓词里的各个公式
            for item in input_str:
                tmp += item
                # 分隔
                if item == '(' or item == ',' or item == ')':
                    self.data.append(tmp[0:len(tmp)-1])
                    tmp = ""
        # data第一项是谓词，后面是谓词里的各个公式
        # print(self.data)

    # 查看是否为负公式，即有无 "¬" 前缀
    def is_negative(self):
        return self.data[0][0] == "¬"

    # 返回谓词名称
    def get_name(self):
        if self.is_negative():
            return self.data[0][1:]
        else:
            return self.data[0]

    # 返回整个谓词公式
    def get_item(self):
        # tmp = 谓词 + 谓词的左括号
        tmp = self.data[0] + "("
        # 谓词里的公式
        for j in range(1, len(self.data)):
            tmp = tmp + self.data[j]
            # 添加逗号
            if j < len(self.data) - 1:
                tmp = tmp + ","
        # tmp += 谓词对应的右括号
        tmp += ")"
        return tmp

    # 变量更名
    def rename(self, old_name, new_name):
        for i in range(len(old_name)):
            for j in range(1, len(self.data)):
                if self.data[j] == old_name[i]:
                    self.data[j] = new_name[i]


#  判断两个谓词是否一致
def equal(a: WeiCi, b: WeiCi):
    flag = False
    for i in range(min(len(a.data), len(b.data))):
        flag |= (a.data[i] == b.data[i])
    return flag


# 判断谓词 a 是否在 谓词列表 b 中
def in_data(a: WeiCi, b: list[WeiCi]):
    for item in b:
        if equal(a, item):
            return True
    return False
