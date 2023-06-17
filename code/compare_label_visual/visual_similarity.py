from compare_label_visual import my_io
import numpy as np
import math

# 计算两个向量的欧氏距离
def calculate_euclidean(aim_picture, one_demo):
    dist = np.sqrt(np.sum(np.square(aim_picture - one_demo)))
    return dist

# 计算待完备图像与4500张图片的视觉相似度
def get_visual_similarity(all_vector, test_vector):
    # 处理待完备图像已经生成的特征向量
    test_vector = np.array(test_vector)
    test_vector = test_vector.astype('float64')
    # 处理整个图片库已经生成的特征矩阵
    all_vector = np.array(all_vector)
    all_vector = all_vector.astype('float64')
    # 计算待完备图像和4500张图片的视觉相似度
    visual_similarity = []  # 1*4500
    for i in range(0, len(all_vector)):
        dist = calculate_euclidean(test_vector, all_vector[i])
        visual_similarity.append(math.exp(-dist))
    # 数据规范化
    visual_similarity = (np.array(visual_similarity)-min(visual_similarity))/(max(visual_similarity)-min(visual_similarity))
    return np.array(visual_similarity)
