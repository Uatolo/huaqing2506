# 变量类型
name = "Tom"  # str
age = 22        # int
grades = [95, 85, 87]  # list
info = {"name": "Tom", "age": 22}  # dict

# 类型转换
age_str = str(age)
number = int("123")

# 作用域
x = 5  # 全局变量
def my_function():
    y = 1  # 局部变量
    global x
    x += 1
    print(f"Inside function: x={x}, y={y}")

my_function()
print(f"Outside function: x={x}")