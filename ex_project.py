# 중식당

print('안녕하세요 ㅇㅇ 반점입니다')
menu1 = ['짬뽕', 5000, 3]         # 메뉴명, 가격, 수량
menu2 = ['짜장', 5000, 3]
menu3 = ['탕수육', 10000, 3]

totalPrice = 0
tryNum = 0

# def menufunc(menu):
#     print("{}을 선택하셨습니다".format(menu[0]))
#     choiceMenu = menu[0]
#     choicePrice = menu[1]
#     num = int(input("수량을 입력하세요 : "))
#     while num > menu[2]:
#         print('현재 재고 : {}'.format(menu[2]))
#         num = int(input('남은 재고 내에서 다시 수량을 입력해주세요'))
#     return choiceMenu, choicePrice, num

while True:
    print("\n")
    print("*" * 27)
    print("* 1. {} (재고수량 : {})   *".format(menu1[0], menu1[2]))
    print("* 2. {} (재고수량 : {})   *".format(menu2[0], menu2[2]))
    print("* 3. {} (재고수량 : {}) *".format(menu3[0], menu3[2]))
    print("*" * 27, "\n")

    while True :
        choice = input("원하는 메뉴를 선택하세요 : ")
        if choice == '1' or choice == "짬뽕":
            print(menu1[0] + "을 선택하셨습니다")
            choiceMenu = menu1[0]
            choicePrice = menu1[1]
            num = int(input('수량을 입력하세요 : '))
            while num > menu1[2]:
                print('현재 재고 : {}'.format(menu1[2]))
                num = int(input("남은 재고 내에서 다시 수량을 입력해주세요 : "))

            # menufunc(menu1)
            menu1[2] -= num
            totalPrice += choicePrice * num
            print('현재 결제금액은', totalPrice, '입니다.')
            break
        elif choice == '2' or choice == "짜장":
            print(menu2[0] + "을 선택하셨습니다")
            choiceMenu = menu2[0]
            choicePrice = menu2[1]
            num = int(input('수량을 입력하세요 : '))
            while num > menu1[2]:
                print('현재 재고 : {}'.format(menu2[2]))
                num = int(input("남은 재고 내에서 다시 수량을 입력해주세요 : "))
            menu2[2] -= num
            totalPrice += choicePrice * num
            print('현재 결제금액은', totalPrice, '입니다.')
            break
        elif choice == '3' or choice == "탕수육":
            print(menu3[0] + '을 선택하셨습니다')
            choiceMenu = menu3[0]
            choicePrice = menu3[1]
            num = int(input('수량을 입력하세요 : '))
            while num > menu1[2]:
                print('현재 재고 : {}'.format(menu3[2]))
                num = int(input("남은 재고 내에서 다시 수량을 입력해주세요 : "))
            menu3[2] -= num
            totalPrice += choicePrice * num
            print('현재 결제금액은', totalPrice, '입니다.')
            break
        else:
            print("1~3 의 숫자 혹은 정확한 메뉴 이름을 다시 선택해주세요")
            tryNum += 1
            if tryNum >= 5:
                print("NAGA!")
                break

    if tryNum >= 5:
        break

    contin = input("주문을 계속하시겠습니까? : ")
    if contin == 'yes' or contin == "예":
        continue
    else:
        break

print('총 결제금액은', totalPrice, '입니다.')
