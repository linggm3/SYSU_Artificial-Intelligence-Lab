from Class import WeiCi


# print子句
def print_clause(clause):
    tmp = '('
    # 加上子句的各个谓词
    for i in range(len(clause)):
        # 加上谓词
        tmp += clause[i].get_item()
        # 添加逗号
        if i < len(clause) - 1:
            tmp = tmp + ","
    # 添加最外层右括号
    tmp += ')'
    print(tmp)


# 展示推理过程
def show_process(key, i, j, old_name, new_name, set_of_clause):
    # 输出归结过程，如 R[1, 2a](x=sue) = (Student(sue))
    tmp = str(len(set_of_clause)) + ": R[" + str(i + 1)
    if len(new_name) == 0 and len(set_of_clause[i]) != 1:
        # 'a'的 ascii 码为 97
        tmp = tmp + chr(key + 97)
    tmp = tmp + ", " + str(j + 1) + chr(key + 97) + "]("
    # unify，归结改名展示
    for k in range(len(old_name)):
        tmp = tmp + old_name[k] + "=" + new_name[k]
        if k < len(old_name) - 1:
            tmp = tmp + ", "
    tmp = tmp + ") = "
    print(tmp, end="")
