# 중식당

print('안녕하세요 ㅇㅇ 반점입니다')
menu1 = ['짬뽕', 5000]
menu2 = ['짜장', 5000]
menu3 = ['탕수육', 10000]

print("*"*27)
print("*", '1.', menu1[0], '2.', menu2[0], '3.', menu3[0], '*')
print("*"*27)

tryNum = 0
while True :
    choice = input("원하는 메뉴를 선택하세요 : ")

    if choice == '1' or choice == "짬뽕":
        print(menu1[0] + "을 선택하셨습니다")
        choice = menu1
        break
    elif choice == '2' or choice == "짜장":
        print(menu2[0] + "을 선택하셨습니다")
        choice = menu2
        break
    elif choice == '3' or choice == "탕수육":
        print(menu3[0] + '을 선택하셨습니다')
        choice = menu3
        break
    else:
        print("1~3 의 숫자 혹은 정확한 메뉴 이름을 다시 선택해주세요")
        tryNum += 1
        if tryNum >= 5 :
            print("NAGA!")
            break

num = int(input('수량을 입력하세요 : '))
print('총 결제금액은', choice[1] * num, '입니다.')
