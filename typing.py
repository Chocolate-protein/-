ques = '동해물과 백두산이 마르고 닳도록'

while True:
    print(ques)
    ans = input()
    worng = ''

    if len(ques) > len(ans):
        ans += ' ' * (len(ques)-len(ans))

    if ques == ans:
        print('정답입니다')
        break
    else:
        print("오답입니다")
        for i in range(len(ans)):
            if ans[i] == ques[i]:
                worng += ans[i]
                continue
            else:
                worng += "X"
        print("틀린 부분 : {}".format(worng))
        print("다시 입력하세요\n")