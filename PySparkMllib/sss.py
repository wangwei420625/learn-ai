liuOld = 10
guanOld = 10
zhangOld = 10

num = 0
while True:
    num += 1
    liuNew = guanOld
    guanNew = liuOld/2.0 + zhangOld
    zhangNew = liuOld/2.0

    if liuOld == liuNew and guanOld == guanNew and zhangOld == zhangNew:
        print(num, liuNew, guanNew, zhangNew)
        break
    liuOld, guanOld, zhangOld = liuNew, guanNew, zhangNew
