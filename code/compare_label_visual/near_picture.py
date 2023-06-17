import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from compare_label_visual import my_io, visual_similarity, label_similarity
from do_sift import sift_process
from do_cnn import cnn_process
from picture_label import label
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 近邻检索
class get_near_picture(QThread):
    signal_result = pyqtSignal(list, list)
    signal_str = pyqtSignal(str)

    def __init__(self, test_path, img_img_visual_weight, num, sift_cnn_weight, all_sift, all_cnn, test_sift, test_cnn):
        super(get_near_picture, self).__init__(parent=None)
        self.test_path = test_path
        self.img_img_visual_weight = img_img_visual_weight
        self.num = num
        self.sift_cnn_weight = sift_cnn_weight
        self.all_vector_sift = all_sift
        self.all_vector_cnn = all_cnn
        self.test_vector_sift = test_sift
        self.test_vector_cnn = test_cnn
        self.context = ""

    def run(self):
        self.signal_str.emit("正在计算视觉相似度")
        visualSimilarity_sift = visual_similarity.get_visual_similarity(self.all_vector_sift, self.test_vector_sift)
        visualSimilarity_cnn = visual_similarity.get_visual_similarity(self.all_vector_cnn, self.test_vector_cnn)
        visualSimilarity = visualSimilarity_sift * self.sift_cnn_weight + visualSimilarity_cnn * (1-self.sift_cnn_weight)
        visualSimilarity = (visualSimilarity-min(visualSimilarity))/(max(visualSimilarity)-min(visualSimilarity))
        self.signal_str.emit("正在计算语义相似度")
        labelSimilarity = label_similarity.get_label_similarity(self.test_path)
        self.signal_str.emit("正在计算综合相似度")
        similarity = np.array(labelSimilarity) * (1 - self.img_img_visual_weight) + np.array(visualSimilarity) * self.img_img_visual_weight
        similarity = (similarity - min(similarity)) / (max(similarity) - min(similarity))
        self.signal_str.emit("正在查找近邻图像")
        picture_index = list(np.arange(0, 4500, 1))
        temp = dict(zip(picture_index, list(similarity)))  # 生成键值对
        # 找出近邻图像的id
        near_picture_id = []
        for i in range(self.num):
            max_id = max(temp, key=temp.get)
            near_picture_id.append(max_id)
            del temp[max_id]  # 根据key删除键值对
        # 找出近邻图像名称
        near_picture_names = []
        all_name = my_io.readCsv("D:/Python/PyCharm/Projects/Label_System/files/data_files"
                                 "/initial_picture_name.csv", all=True)
        all_name = all_name.squeeze(axis=1)
        for p_id in near_picture_id:
            near_picture_names.append(all_name[p_id])
        context = "\n近邻图像索引及所带标签****************************************>\n"
        for i in range(0, np.size(near_picture_names)):
            context = context + "{0:>3}  : {1}\n".format(i+1, label.get_theInitialPicture_all_label(near_picture_names[i]))
        context += "\n候选标签是: {}\n".format(self.get_label(near_picture_names))
        self.context = context
        self.signal_str.emit("近邻检索完成")
        self.signal_result.emit(list(near_picture_names), list(visualSimilarity))
        return

    def get_label(self, all_near_names):
        # 提取候选标签
        candidate_labels = set()  # 创建空集合，记录候选标签
        all_near_picture_label = []  # 记录每个近邻图像所带有的标签
        for name in all_near_names:
            labs = label.get_theInitialPicture_all_label(name)  # 取得一张待完备图像的候选标签
            all_near_picture_label.append(labs)
            for one_label in labs:
                candidate_labels.add(one_label)  # 将标签加入集合，重复的标签自动忽略
        # 从候选标签中去掉初始标签，得到真正的候选标签集合
        for e in label.get_theTestPicture_all_label(self.test_path):
            candidate_labels.discard(e)  # 从去掉初始标签，不存在不报错
        candidate_labels = list(candidate_labels)
        return candidate_labels

    def stop(self):
        self.terminate()

# 特征提取
class get_visual_feature(QThread):
    signal_str = pyqtSignal(str)
    signal_result = pyqtSignal(list, list, list, list)

    def __init__(self, test_path, cnn_input_imgSize, cnn_cut, sift_cnn_weight):
        super(get_visual_feature, self).__init__(parent=None)
        self.test_path = test_path
        self.cnn_input_imgSize = cnn_input_imgSize  # 元组
        self.cnn_cut = cnn_cut                      # bool
        self.sift_cnn_weight = sift_cnn_weight      # double

    def run(self):
        self.signal_str.emit("正在读取SIFT特征矩阵")
        all_vector_sift = \
            my_io.readCsv("D:/Python/PyCharm/Projects/Label_System/code/do_sift/pca_all_vector_sift.csv", all=True)

        self.signal_str.emit("正在读取CNN特征矩阵")
        all_vector_cnn = \
            my_io.readCsv("D:/Python/PyCharm/Projects/Label_System/code/do_cnn/pca_all_vector_cnn.csv", all=True)

        self.signal_str.emit("正在提取SIFT特征")
        test_vector_sift = sift_process.get_sift_one_vector(self.test_path)

        self.signal_str.emit("正在提取CNN特征")
        test_vector_cnn = cnn_process.get_cnn_one_vector(self.test_path)
        # 发射提取结果
        self.signal_str.emit("特征提取完成")
        self.signal_result.emit(list(all_vector_sift), list(all_vector_cnn),
                                list(test_vector_sift), list(test_vector_cnn))

    def stop(self):
        self.terminate()







# # 测试近邻搜索
# def test(test_picture=1):  # 待完备图像名称
#     # 显示待完备图像
#     test_picture = str(test_picture)
#     path = "D:/Python/PyCharm/Projects/Label_System/files/picture_files/test_pictures/{}.jpeg".format(test_picture)
#     test_label = label.get_theTestPicture_all_label(path)  # 取得标签
#     picture.show_picture(path, test_label)  # 显示图片和标签
#     print("待完备图像 - {}:   ".format(test_picture), test_label)
#     # 查找近邻图像
#     near_picture_name = get_near_picture(
#         test_path=path,  # 待完备图像路径
#         weight_1=0.7,           # 视觉 /（视觉+语义）
#         num=6,                # 显示的近邻图像数量
#         sift=True,
#         cnn=True,
#         label_sim=True,
#         sift_weight=0.4       # sift / (sift + cnn)
#     )
#     # 显示找到的近邻图像
#     for name in near_picture_name:
#         one_label = label.get_theInitialPicture_all_label(name)  # 取得标签
#         picture.show_picture_from_name(name, one_label)  # 显示图片和标签
#         print("近邻图像 - {}:   ".format(name), one_label)
#     return
