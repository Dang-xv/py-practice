import random
def guess_number():
    print("=== 猜数字游戏 ===")
    target = random.randint(1, 100)
    count = 0

    while True:
        try:
            num = int(input("请猜一个 1~100 的数字："))
            count += 1

            if num < target:
                print("小了！再大一点")
            elif num > target:
                print("大了！再小一点")
            else:
                print(f"恭喜猜对！答案是 {target}，你猜了 {count} 次")
                break
        except:
            print("请输入有效数字！")

guess_number()