# -*- coding: utf-8 -*-
# @Author  : Lan Zhang
# @Time    : 2022/3/21 10:04
# @File    : train_test.py
# @Software: PyCharm
import os
import random


def create_val(txt_path):
    fh = open(txt_path, 'r')  # 读取文件
    imgs = []  # 用来存储路径与标签
    # 一行一行的读取
    for line in fh:
        line = line.rstrip()  # 这一行就是图像的路径，以及标签
        words = line.split(' ')
        imgs.append((os.path.join('/home/A/liujing/zhang/DCNN_pre/CASIA_Align', words[0]), int(words[1])))
        # 路径和标签添加到列表中
    random.shuffle(imgs)
    return imgs


imgs = create_val('./data.txt')
pos = int(len(imgs) * 0.8)
train = imgs[:pos]
val = imgs[pos:]
with open('data_train.txt', 'w') as f:
    for line in train:
        f.write(str(line) + '\n')
with open('data_validation.txt', 'w') as ff:
    for line in val:
        ff.write(str(line) + '\n')
print("Done!")
