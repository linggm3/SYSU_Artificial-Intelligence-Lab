from Class import WeiCi
from Class import in_data
from Output import print_clause
from Output import show_process
from Input import input_clauses


# 判断新产生的子句 与 现有子句集中的子句 是否重复
def duplicated_or_not(new_clause, clauses, key, mode):
    # 新子句长度为 1
    if len(new_clause) == 1:
        for k in range(len(clauses)):
            if len(clauses[k]) == 1 and new_clause[0].maze_board == clauses[k][0].maze_board:
                if type(key) == int:
                    key = -1
                elif type(key) == list:
                    key = []
                break
    elif mode == 2:
        for clause in clauses:
            flag = True
            for item in new_clause:
                if not in_data(item, clause):
                    flag = False
            if flag:
                return []
    return key


# 判断归结循环是否结束
def end_or_not(new_clause, clauses):
    # 新生成的new_clause已经为空
    if len(new_clause) == 0:
        print("[]")
        return True
    # 查找已有的子句中是否存在与新子句互补
    if len(new_clause) == 1:
        for i in range(len(clauses) - 1):
            if len(clauses[i]) == 1 and new_clause[0].get_name() == clauses[i][0].get_name() and new_clause[0].maze_board[1:]\
                    == clauses[i][0].maze_board[1:] and new_clause[0].is_negative() != clauses[i][0].is_negative():
                print(len(clauses) + 1, ": R[", i + 1, ", ", len(clauses), "]() = []", sep="")
                return True
    return False


# 单步归结过程
def unify(clause1, clause2, clauses):
    # 将自由变量更名为约束变量
    old_name = []
    new_name = []
    # pos == -1 表示该子句的同名谓词不能进行消去
    pos = -1
    # 在子句 clause2 中找相同的谓词，且可以消去，设置pos为其位置
    for k in range(len(clause2)):
        # 谓词名字一样，而且是互补项
        if clause1[0].get_name() == clause2[k].get_name() and clause1[0].is_negative() != clause2[k].is_negative():
            pos = k
            # 找到可以换名的变量并记录
            for l in range(len(clause2[k].maze_board) - 1):
                # 是自由变量（设定 只有一个小写字母的公式 为 变量）
                if len(clause2[k].maze_board[l + 1]) == 1:
                    old_name.append(clause2[k].maze_board[l + 1])
                    new_name.append(clause1[0].maze_board[l + 1])
                elif len(clause1[0].maze_board[l + 1]) == 1:
                    old_name.append(clause1[k].maze_board[l + 1])
                    new_name.append(clause2[0].maze_board[l + 1])
                elif clause2[k].maze_board[l + 1] != clause1[0].maze_board[l + 1]:
                    pos = -1
                    break
            break
    # 两个子句无法进行归结
    if pos == -1:
        control_flag = 'continue'
        return control_flag, clauses, False
    # 可以归结，改名，消去互补项，生成新子句
    # 记录生成的新子句
    new_clause = []
    for k in range(len(clause2)):
        # 位置为 pos 的已经被消去了，所以不在新子句里
        if k != pos:
            p = WeiCi("")
            # 往 p 里添加公式
            for item in clause2[k].maze_board:
                p.data.append(item)
            p.rename(old_name, new_name)
            new_clause.append(p)
    pos = duplicated_or_not(new_clause, clauses, pos, 1)
    # 如果生成的子句已存在，跳过加入子句集的过程
    if pos == -1:
        control_flag = 'continue'
        return control_flag, clauses, False
    # 生成的新的子句加入的子句集中
    clauses.append(new_clause)
    # 展示归结过程
    show_process(pos, clauses.index(clause1), clauses.index(clause2), old_name, new_name, clauses)  # 输出生成新子句的相关信息
    # 输出该新子句
    print_clause(new_clause)
    # 判断是否应该结束归结过程
    if end_or_not(new_clause, clauses):
        control_flag = 'break'
        return control_flag, clauses, True
    return 'success', clauses, True


# 单步归结过程
def unify2(clause1, clause2, clauses):
    counter = len(clauses) + 1
    # 将自由变量更名为约束变量
    old_name = []
    new_name = []
    pos1 = []
    pos2 = []
    # 在 两个子句 中找相同的谓词，且可以消去，设置pos为其位置
    for i in range(len(clause1)):
        for j in range(len(clause2)):
            # print(i, j)
            # 谓词名字一样，而且是互补项
            if clause1[i].get_name() == clause2[j].get_name() and clause1[i].is_negative() != clause2[j].is_negative():
                pos1.append(i)
                pos2.append(j)
                # 找到可以换名的变量并记录
                for l in range(len(clause2[j].maze_board) - 1):
                    # 是自由变量（设定 只有一个小写字母的公式 为 变量）
                    if len(clause2[j].maze_board[l + 1]) == 1 and len(clause1[i].maze_board[l + 1]) != 1:
                        old_name.append(clause2[j].maze_board[l + 1])
                        new_name.append(clause1[i].maze_board[l + 1])

                    elif len(clause1[i].maze_board[l + 1]) == 1 and len(clause2[j].maze_board[l + 1]) != 1:
                        old_name.append(clause1[j].maze_board[l + 1])
                        new_name.append(clause2[i].maze_board[l + 1])

                    elif clause2[j].maze_board[l + 1] != clause1[i].maze_board[l + 1]:
                        pos1.pop()
                        pos2.pop()
                        break
    # 两个子句无法进行归结
    if not pos1:
        control_flag = 'continue'
        return control_flag, clauses, False
    # 可以归结，改名，消去互补项，生成新子句
    # 记录生成的新子句
    new_clause = []
    # print(pos1, pos2)
    for i in range(len(clause1)):
        # 位置为 pos1 的已经被消去了，所以不在新子句里
        if i not in pos1:
            p = WeiCi("")
            # 往 p 里添加公式
            for item in clause1[i].maze_board:
                p.data.append(item)
            p.rename(old_name, new_name)
            new_clause.append(p)
    for j in range(len(clause2)):
        # 位置为 pos 的已经被消去了，所以不在新子句里
        if j not in pos2:
            p = WeiCi("")
            # 往 p 里添加公式
            for item in clause2[j].maze_board:
                p.data.append(item)
            p.rename(old_name, new_name)

            # 避免重复
            if not in_data(p, new_clause):
                new_clause.append(p)

    pos1 = duplicated_or_not(new_clause, clauses, pos1, 2)
    # 如果生成的子句已存在，跳过加入子句集的过程

    if len(pos1) == 0:
        control_flag = 'continue'
        return control_flag, clauses, False
    # 生成的新的子句加入的子句集中
    clauses.append(new_clause)
    # 展示归结过程
    # show_process(0, clauses.index(clause1), clauses.index(clause2), old_name, new_name, clauses)  # 输出生成新子句的相关信息
    print(counter, ': R[', clauses.index(clause1)+1, ', ', clauses.index(clause2)+1, '] = ', end='', sep='')
    # 输出该新子句
    print_clause(new_clause)
    # 判断是否应该结束归结过程
    if end_or_not(new_clause, clauses):
        control_flag = 'break'
        return control_flag, clauses, True
    return 'success', clauses, True


# 一阶谓词逻辑归结过程
def solve(clauses):
    # 标记 继续归结与否
    flag = True
    # 标记 这轮循环有没有生成新子句
    add_flag = True
    while flag:
        if not add_flag:
            break
        # 循环刚开始，设定该轮循环还没有产生新子句
        add_flag = False
        # 子句1
        for i in range(len(clauses)):
            if not flag:
                break
            if len(clauses[i]) == 1:
                # 子句2
                for j in range(0, len(clauses)):
                    if not flag:
                        break
                    if i == j:
                        continue
                    # 单步归结
                    [ctrl_flag, clauses, tmp_flag] = unify(clauses[i], clauses[j], clauses)
                    # 如果一个归结循环中 归结出一条新子句 则 add_flag 为真
                    add_flag = add_flag or tmp_flag
                    if ctrl_flag == 'continue':
                        continue
                    elif ctrl_flag == 'break':
                        flag = False
                        break
        # 应用 unify1 无法产生新子句， 则应用 unify2
        if not add_flag:
            # print(add_flag)
            for i in range(len(clauses)):
                if not flag:
                    break
                # 子句2
                for j in range(i+1, len(clauses)):
                    if not flag:
                        break
                    # 单步归结
                    [ctrl_flag, clauses, tmp_flag] = unify2(clauses[i], clauses[j], clauses)
                    # print(tmp_flag)
                    # 如果一个归结循环中 归结出一条新子句 则 add_flag 为真
                    add_flag = add_flag or tmp_flag
                    # print(add_flag)
                    if ctrl_flag == 'continue':
                        continue
                    elif ctrl_flag == 'break':
                        flag = False
                        break
        if not add_flag:
            break

    if add_flag:
        print("成功归结出NIT!")
    else:
        print("无新子句产生!")


if __name__ == '__main__':
    cs = input_clauses()
    solve(cs)
