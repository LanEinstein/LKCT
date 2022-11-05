import os
"""
本文件可以依据路径文件（.txt格式）创建与原数据集相同的外层文件结构。
"""


fh = open(r'F:\casia_new\clean_data.txt', 'r')  # 读取文件
imgs = []  # 用来存储路径与标签
# 一行一行的读取
for line in fh:
    line = line.rstrip()  # 这一行就是图像的路径，以及标签
    words = line.split(' ')
    imgs.append(
        (os.path.join(r'F:\casia_new\CASIA-maxpy-clean', words[0])))  # 路径添加到列表中
for line in imgs:
    the_folder = line.split('\\')[-2]
    root_path = r'F:\CASIA_Align'
    full_path = os.path.join(root_path, the_folder)
    if os.path.exists(full_path) is not True:
        os.mkdir(full_path)
        print('-->', the_folder)
    else:
        continue
