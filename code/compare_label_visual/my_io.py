# -*- coding:utf-8- -*-
import numpy as np
import csv


# 保存数据到CSV文件
def saveCsv(path, content):
    try:
        np.savetxt(path, content, fmt='%.16f', delimiter=',')
        return True
    except Exception:
        print("保存出现异常！")
        return False

# 读取CSV文件
def readCsv(path, all=True, H=0, L=0,):
    content = []
    with open(path) as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            content.append(row)
    content = np.array(content)
    if all:
        return content
    else:
        return content[H][L]