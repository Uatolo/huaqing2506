import os
import shutil
from sklearn.model_selection import train_test_split
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 数据集路径
dataset_dir = r'D:\pythonproject\shixun\DAY3\Images'  # 替换为你的数据集路径
train_dir = r'D:\pythonproject\shixun\DAY3\Images\train'  # 训练集输出路径
val_dir = r'D:\pythonproject\shixun\DAY3\Images\val'  # 验证集输出路径

# 划分比例
train_ratio = 0.7

# 创建训练集和验证集目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历每个类别文件夹
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)

    # 跳过已创建的train/val目录和非目录文件
    if class_name in ["train", "val"] or not os.path.isdir(class_path):
        continue

    # 获取该类别下的所有图片（使用完整路径）
    images = [os.path.join(class_path, f)
              for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 划分训练集和验证集
    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)

    # 创建类别子文件夹
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # 移动训练集图片
    for src in train_images:
        dst = os.path.join(train_class_dir, os.path.basename(src))
        shutil.move(src, dst)

    # 移动验证集图片
    for src in val_images:
        dst = os.path.join(val_class_dir, os.path.basename(src))
        shutil.move(src, dst)

    # 删除空类别文件夹
    try:
        shutil.rmtree(class_path)
    except OSError as e:
        print(f"无法删除目录 {class_path}: {e.strerror}")