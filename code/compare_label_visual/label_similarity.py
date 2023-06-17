import numpy as np
from compare_label_visual import my_io

all_picture_nums = 4500
entrance_ex_set = 0.7

# 计算两个标签的共现频率
def get_both_label_co(aim_id, one_id, all_initial_label):
    # 读取标签矩阵
    all_initial_label = np.transpose(all_initial_label)
    temp_vector = all_initial_label[aim_id] + all_initial_label[one_id]
    up = list(temp_vector).count(2)
    down_1 = list(all_initial_label[aim_id]).count(1)
    down_2 = list(all_initial_label[one_id]).count(1)
    return (up / down_1 + up / down_2) / 2

# 构建标签间接相关集
def get_expanded_labelSet(label_id, all_initial_vector):
    all_initial_vector = np.transpose(all_initial_vector)
    label_vector = all_initial_vector[label_id]  # label_id所在向量
    label_id_count = list(label_vector).count(1)
    expanded_labelSet = []  # 记录扩展的相关标签
    for i in range(0, np.shape(all_initial_vector)[0]):
        if i != label_id:
            temp = list(label_vector + all_initial_vector[i])
            if temp.count(2)/label_id_count > entrance_ex_set:  # 某共现频率大于entrance_ex_set
                expanded_labelSet.append(i)  # 记录标签id
    return expanded_labelSet  # 记录了间接相关度的标签id

# 求两个标签的谷歌距离
def get_two_label_google_dist(the_label_id, another_label_id, all_label_vector):
    all_label_vector = np.transpose(all_label_vector)
    the_label_id_count = np.sum(all_label_vector[the_label_id])
    if the_label_id_count == 0:
        the_label_id_count = 0.000000001
    another_label_id_count = np.sum(all_label_vector[another_label_id])
    if another_label_id_count == 0:
        another_label_id_count = 0.000000001
    both_label_id_count = 0
    for e in (np.array(all_label_vector[the_label_id])+np.array(all_label_vector[another_label_id])):
        if e == 2:
            both_label_id_count += 1
    if both_label_id_count == 0:
        both_label_id_count = 0.000000001
    up = max(np.log10(the_label_id_count), np.log10(another_label_id_count) - np.log10(both_label_id_count))
    down = np.log10(all_picture_nums) - min(np.log10(the_label_id_count), np.log10(another_label_id_count))
    dist = up / down  # 得到两标签的google距离
    return dist

# 计算两个标签的直接相关度
def get_two_label_direct_sim(the_label_id, another_label_id, all_label_vector):
    dist = get_two_label_google_dist(the_label_id, another_label_id, all_label_vector)
    return np.exp(-dist)

# 计算两个标签的间接相关度
def get_two_label_indirect_sim(the_label_id, another_label_id, all_label_vector):
    # 1.分别构建两个标签的扩展标签集
    ex_labet_set_1 = get_expanded_labelSet(the_label_id, all_label_vector)
    ex_labet_set_2 = get_expanded_labelSet(another_label_id, all_label_vector)
    # 2.返回与扩展标签集中标签直接相关度的最大值
    indirect_sim_list = [0, 0]
    if ex_labet_set_1:  # 扩展标签集1不为空
        sim = []
        for indirect_id in ex_labet_set_1:
            sim.append(get_two_label_direct_sim(indirect_id, another_label_id, all_label_vector))
        indirect_sim_list[0] = max(sim)
    if ex_labet_set_2:  # 扩展标签集2不为空
        sim = []
        for indirect_id in ex_labet_set_2:
            sim.append(get_two_label_direct_sim(the_label_id, indirect_id, all_label_vector))
        indirect_sim_list[1] = max(sim)
    # 3.求平均
    indirect_sim = np.sum(indirect_sim_list)/2  # 两标签的间接相关度
    return indirect_sim

# 计算两标签的最终语义相关度
def get_two_label_final_sim(the_id, another_id, all_label_vector):
    direct_sim = get_two_label_direct_sim(the_id, another_id, all_label_vector)
    # indirect_sim = get_two_label_indirect_sim(the_id, another_id, all_label_vector)
    # return (direct_sim + indirect_sim)/2
    return direct_sim

# 计算sigmoid权重
def get_sigmoid_weight(aim_id, all_label_vector):
    all_label_vector = np.transpose(all_label_vector)
    count = list(all_label_vector[aim_id]).count(1)
    sigmoid_weight = 1 / (1 + (count / all_picture_nums))
    return sigmoid_weight

# 返回待完备图像和某一张图的语义相关度（初始标列号，某一张图所带标签列号，初始标签矩阵）
def get_bothPicture_label_similarity(aim_picture_col_id, one_demo_col_id, initial_all_label):
    up = 0
    for id_1 in aim_picture_col_id:
        # 加入初始标签的sigmoid权重
        sigmoid_weight = get_sigmoid_weight(id_1, initial_all_label)
        for id_2 in one_demo_col_id:
            up += get_two_label_final_sim(id_1, id_2, initial_all_label) * sigmoid_weight
    down = np.size(aim_picture_col_id) * np.size(one_demo_col_id)
    label_similarity = up/down
    return label_similarity

# 返回待完备图像和所有图像的语义相关度
def get_label_similarity(test_path):
    # 解析路径，取出待完备图片名称
    temp = str(test_path).split('/')
    test_name = str(str(temp[np.size(temp)-1]).split('.')[0])
    # 获取待完备图像的标签向量
    test_all_name = my_io.readCsv(
        "D:/Python/PyCharm/Projects/Label_System/files/data_files/test_picture_name.csv", all=True)
    test_all_name = test_all_name.squeeze(axis=1)
    aim_picture_id = list(test_all_name).index(test_name)
    test_all_label = my_io.readCsv(
        "D:/Python/PyCharm/Projects/Label_System/files/data_files/test_picture_label_eigenmatrix.csv", all=True)
    aim_label_vector = test_all_label[aim_picture_id]
    aim_picture_col_id = []
    for i in range(np.size(aim_label_vector)):  # 找出待完备图片所对应标签的列索引
        if aim_label_vector[i] == '1':
            aim_picture_col_id.append(i)
    # 处理待完备图像和训练集图片的关系
    initial_all_label = my_io.readCsv(
        "D:/Python/PyCharm/Projects/Label_System/files/data_files/initial_picture_label_eigenmatrix.csv", all=True)
    initial_all_label = initial_all_label.astype('int')
    label_similarity = []  # 1*4500
    for count in range(4500):
        # 获取训练集中某图片的标签
        one_demo_vector = initial_all_label[count]
        one_demo_col_id = []
        for i in range(np.size(one_demo_vector)):  # 找出训练集中某图片所对应标签的列索引
            if one_demo_vector[i] == 1:
                one_demo_col_id.append(i)
        label_similarity.append(get_bothPicture_label_similarity(aim_picture_col_id, one_demo_col_id, initial_all_label))
    # 规范化处理
    label_similarity = (np.array(label_similarity)-min(label_similarity))/(max(label_similarity)-min(label_similarity))
    return label_similarity

