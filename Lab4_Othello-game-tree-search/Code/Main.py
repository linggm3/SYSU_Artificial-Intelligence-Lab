import time
from Class import Chess

start = time.time()

weight = [[293.1687192717666], [-18.445045191386637, -67.09295838393061], [39.3803316628712, -62.281630874386494, -10.387943568777242], [40.021627067404765, -4.499765006227619, 13.534016195908379, 1.797975550196604], 72.05120882477823, 0]
chess1 = Chess(weight)
print("选择先手请输入1，选择后手请输入-1")
chose = int(input())
for epoch in range(30):
    if chose == -1:
        [res, pos] = chess1.minimax_search_with_abcut(epoch, epoch + 3, 1, show=False)
        print('\n', res, pos)
        if len(pos) != 0:
            chess1.drops(pos[0], pos[1], 1)
        chess1.show_board(5, title='score: ' + str(int(chess1.final_score()/10000)))

        valid = False
        while not valid:
            r = int(input('请输入行 '))
            c = int(input('请输入列 '))
            valid = chess1.drops(r, c, -1)[0]
            if not valid:
                if chess1.has_valid(-1):
                    print('非法落子，请重新落子')
                else:
                    print('无子可走')
                    break
        chess1.show_board(5, title='score: ' + str(int(chess1.final_score() / 10000)))
    else:
        chess1.show_board(5, title='score: ' + str(int(chess1.final_score() / 10000)))
        valid = False
        while not valid:
            r = int(input('请输入行 '))
            c = int(input('请输入列 '))
            valid = chess1.drops(r, c, 1)[0]
            if not valid:
                if chess1.has_valid(1):
                    print('非法落子，请重新落子')
                else:
                    print('无子可走')
                    break

        chess1.show_board(5, title='score: ' + str(int(chess1.final_score() / 10000)))

        [res, pos] = chess1.minimax_search_with_abcut(epoch, epoch + 3, -1, show=False)
        print('\n', res, pos)
        if len(pos) != 0:
            chess1.drops(pos[0], pos[1], -1)


final_score = chess1.final_score()
chess1.show_board(time=5, title='score: '+str(int(chess1.final_score()/10000)))

end = time.time()
print("共用时: ", end-start)