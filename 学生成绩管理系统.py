import json

students = []
filename = "students.json"


def load_data():
    global students
    try:
        with open(filename, "r", encoding="utf-8") as f:
            students = json.load(f)
    except:
        students = []


def save_data():
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(students, f, ensure_ascii=False, indent=2)


def show_menu():
    print("=" * 50)
    print("        学生成绩管理系统（增强版）")
    print("1. 添加学生")
    print("2. 删除学生")
    print("3. 修改学生成绩")
    print("4. 查询学生")
    print("5. 显示所有学生")
    print("6. 按总分排序排名")
    print("7. 统计总分与平均分")
    print("0. 退出系统")
    print("=" * 50)


def add_student():
    sid = input("请输入学号：")
    for stu in students:
        if stu["id"] == sid:
            print("该学号已存在！")
            return
    name = input("请输入姓名：")
    try:
        chinese = float(input("语文成绩："))
        math = float(input("数学成绩："))
        english = float(input("英语成绩："))
    except:
        print("成绩必须为数字！")
        return

    stu = {
        "id": sid,
        "name": name,
        "chinese": chinese,
        "math": math,
        "english": english
    }
    students.append(stu)
    save_data()
    print("添加成功！")


def delete_student():
    sid = input("请输入要删除的学号：")
    for stu in students:
        if stu["id"] == sid:
            students.remove(stu)
            save_data()
            print("删除成功！")
            return
    print("未找到该学生")


def modify_student():
    sid = input("请输入要修改的学号：")
    for stu in students:
        if stu["id"] == sid:
            try:
                stu["chinese"] = float(input("新语文："))
                stu["math"] = float(input("新数学："))
                stu["english"] = float(input("新英语："))
            except:
                print("输入无效")
                return
            save_data()
            print("修改成功！")
            return
    print("未找到该学生")


def search_student():
    sid = input("请输入学号：")
    for stu in students:
        if stu["id"] == sid:
            total = stu["chinese"] + stu["math"] + stu["english"]
            print("\n学号：%s  姓名：%s" % (stu["id"], stu["name"]))
            print("语文：%.1f  数学：%.1f  英语：%.1f  总分：%.1f" %
                  (stu["chinese"], stu["math"], stu["english"], total))
            return
    print("未找到该学生")


def show_all():
    if not students:
        print("暂无学生")
        return
    print("\n%-10s %-6s %-6s %-6s %-6s %-6s" % ("学号", "姓名", "语文", "数学", "英语", "总分"))
    print("-" * 50)
    for stu in students:
        total = stu["chinese"] + stu["math"] + stu["english"]
        print("%-10s %-6s %-6.1f %-6.1f %-6.1f %-6.1f" %
              (stu["id"], stu["name"], stu["chinese"], stu["math"], stu["english"], total))


def sort_by_total():
    if not students:
        print("暂无学生")
        return
    sorted_stus = sorted(students, key=lambda x: x["chinese"] + x["math"] + x["english"], reverse=True)
    print("\n【按  总分降序排名】")
    print("%-4s %-10s %-6s %-6s" % ("排名", "学号", "姓名", "总分"))
    print("-" * 40)
    for i, stu in enumerate(sorted_stus, 1):
        total = stu["chinese"] + stu["math"] + stu["english"]
        print("%-4d %-10s %-6s %-6.1f" % (i, stu["id"], stu["name"], total))


def statistic():
    if not students:
        print("暂无学生")
        return
    count = len(students)
    total_sum = 0
    for stu in students:
        total_sum += stu["chinese"] + stu["math"] + stu["english"]
    avg = total_sum / count
    print(f"总人数：{count}")
    print(f"所有学生总分和：{total_sum:.1f}")
    print(f"人均总分：{avg:.1f}")


def main():
    load_data()
    while True:
        show_menu()
        c = input("请选择：")
        if c == "1":
            add_student()
        elif c == "2":
            delete_student()
        elif c == "3":
            modify_student()
        elif c == "4":
            search_student()
        elif c == "5":
            show_all()
        elif c == "6":
            sort_by_total()
        elif c == "7":
            statistic()
        elif c == "0":
            print("已保存数据，欢迎下次使用！")
            break
        else:
            print("输入错误")
        input("\n按回车继续...")


if __name__ == "__main__":
    main()
