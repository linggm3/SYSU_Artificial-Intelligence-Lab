from Class import WeiCi
from Output import print_clause


def input_clauses():
    clauses = []
    print("请输入子句数量：")
    clauses_num = input()
    print("请输入", clauses_num, "条子句：")
    for i in range(int(clauses_num)):
        clauses.append([])
        input_clause = input()
        if input_clause == "":
            print("输入空子句，无效")
            return

        if input_clause[0] == '(':
            # 去除子句最外侧的括号
            input_clause = input_clause[1:len(input_clause) - 1]
        # 去除空格
        input_clause = input_clause.replace(' ', '')
        # print(input_clause)
        tmp = ""
        # 将 input_clause 更新到子句集中
        for j in range(len(input_clause)):
            tmp += input_clause[j]
            # 每个谓词公式都以')'结尾，所以按此分割
            if input_clause[j] == ')':
                # print(tmp)
                clause_tmp = WeiCi(tmp)
                # 加入到子句集中
                clauses[i].append(clause_tmp)
                tmp = ""

    for i in range(len(clauses)):
        print(str(i+1) + ': ', end='')
        print_clause(clauses[i])

    return clauses
