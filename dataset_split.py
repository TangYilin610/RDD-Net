import os
import pandas as pd
import random
from collections import defaultdict

# # 读取标签文件
# labels = pd.read_csv('E:/tangyilin/data/Paper_two/Source_domain/Harvard_all.csv')  # 例如，imgname, label
#
# # 获取图像文件夹中的所有图像文件
# image_folder = 'E:/tangyilin/data/Paper_two/Source_domain/Harvard_data/'  # 替换为你的图像文件夹路径
# image_files = os.listdir(image_folder)
#
# # 将图像文件名与标签文件进行匹配
# data = []
#
# for image_file in image_files:
#     # labels['data'] = labels['data'].apply(lambda x: x + '.jpg')
#     # img_name = image_file.split('.')[0]  # 提取图像文件名（去除扩展名）
#     image_file_with_extension = image_file + '.jpg'
#     label = labels.loc[labels['data'] == image_file_with_extension]['glaucoma'].values[0]  # 获取对应标签.loc[sample_idx, 'label'] ，放入原本标签名
#     data.append({'image': image_file_with_extension, 'label': label})
#
# # 创建每个标签对应的图像文件列表的字典
# label_to_images = defaultdict(list)
# for item in data:
#     label_to_images[item['label']].append(item['images']) #这一行是暂时储存的列表名，修改别的数据集时不用修改这一行
#
# # 初始化训练集和测试集
# train = []
# test = []
#
# # 按标签等比例划分数据集
# for label, images in label_to_images.items():
#     num_images = len(images)
#     num_train = int(0.7 * num_images)  # 70%训练集，30%测试集
#     random.shuffle(images)  # 随机打乱顺序
#
#     train.extend([{'ImageName': image, 'glaucoma': label} for image in images[:num_train]])
#     test.extend([{'ImageName': image, 'glaucoma': label} for image in images[num_train:]])
#
# # 创建DataFrame
# train_df = pd.DataFrame(train)
# test_df = pd.DataFrame(test)
#
# # 保存训练集和测试集的文件列表及标签，修改为新的文件名
# train_df.to_csv('F:/tangyilin/TriD-main/Train/Harvard_train.csv', index=False)
# test_df.to_csv('F:/tangyilin/TriD-main/Test/Harvard_test.csv', index=False)
# # -------------------------------从csv文件对应找图片文件并保存-------------------------------
# import os
# import csv
# import shutil
#
# # 定义CSV文件和数据文件所在的目录
# csv_file = 'F:/tangyilin/TriD-main/Test/Havard2_test.csv'  # CSV文件名
# data_folder = 'E:/tangyilin/data/Paper_two/Source_domain/Harvard_data'  # 包含数据文件的文件夹路径
# output_folder = 'F:/tangyilin/TriD-main/dataset/Harvard/test'  # 新的文件夹路径，用于存放筛选后的数据文件
#
# # 创建新文件夹
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 读取CSV文件，获取文件名列表
# with open(csv_file, 'r') as file:
#     reader = csv.reader(file)
#     next(reader)  # 跳过CSV文件的标题行
#     for row in reader:
#         file_name = row[0]  # 假设文件名在CSV文件的第一列
#
#         # 检查数据文件是否存在于数据文件夹中
#         source_file = os.path.join(data_folder, file_name)
#
#         if os.path.exists(source_file):
#             # 复制数据文件到新的文件夹中
#             destination_file = os.path.join(output_folder, file_name)
#             shutil.copyfile(source_file, destination_file)
#         else:
#             print(f"File {file_name} not found in the data folder.")

#
# # #------------------------给列加前缀---------------------------
import pandas as pd

# 读取 CSV 文件
file_path = 'F:/tangyilin/TriD-main/dataset/RIMONE_test.csv'  # 文件路径
column_name = 'ImageName'  # 要修改的列名
prefix = 'RIMONE/test/'  # 要添加的前缀

# 读取 CSV 文件到 DataFrame
data = pd.read_csv(file_path)

# 给指定列的每个元素加上前缀
data[column_name] = prefix + data[column_name].astype(str)

# 将修改后的 DataFrame 保存回 CSV 文件
data.to_csv(file_path, index=False)
#----------------删除列前缀
# import pandas as pd
#
# # 读取 CSV 文件
# file_path = 'F:/tangyilin/TriD-main/Test/Havard_test.csv'
# df = pd.read_csv(file_path)
#
# # 获取要处理的列的名称
# column_name = 'ImageName'
#
# # 删除列中特定前缀
# df[column_name] = df[column_name].str.replace('Harvard/test', '')
# # df[column_name] = df[column_name].str.replace('/', '')
# # 保存修改后的数据为新的 CSV 文件
# new_file_path = 'F:/tangyilin/TriD-main/Test/Havard1_test.csv'
# df.to_csv(new_file_path, index=False)


# # 假设你有一个包含文件名和标签的列表data，形如：data = [{'FileName': 'image1.png', 'Label': 0}, {'FileName': 'image2.png', 'Label': 1}, ...]
# import os
# import csv
#
# old = 'F:/tangyilin/Task-Aug-main/txt_taskaug/test/havardte_list.txt'
# new_folder_path = 'F:/tangyilin/TriD-main/Test/Havard_test.csv'
#
# # 生成 CSV 文件
# with open(new_folder_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['ImageName', 'glaucoma'])
#
#     with open(old, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             image_name = line.strip()[:-1]  # 获取文件名（去除扩展名）
#             label = int(line.strip()[-1])  # 获取最后一个字符作为标签
#             writer.writerow([image_name, label])

#---------------------读取文件--------------------------
# import os
# import csv
#
# # 文件夹路径
# normal_folder = 'F:/tangyilin/TriD-main/dataset/RIMONE/test/normal'
# glaucoma_folder = 'F:/tangyilin/TriD-main/dataset/RIMONE/test/glaucoma'
#
# # 存储文件名和标签的列表
# data = []
#
# # 读取normal文件夹中的文件名并将标签设为0
# for filename in os.listdir(normal_folder):
#     if filename.endswith('.png'):  # 确保只读取图片文件
#         data.append({'ImageName': filename, 'glaucoma': 0})
#
# # 读取glaucoma文件夹中的文件名并将标签设为1
# for filename in os.listdir(glaucoma_folder):
#     if filename.endswith('.png'):  # 确保只读取图片文件
#         data.append({'ImageName': filename, 'glaucoma': 1})
#
# # 将数据写入CSV文件
# csv_file = 'F:/tangyilin/TriD-main/Test/RIMONE_train.csv'
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=['ImageName', 'glaucoma'])
#     writer.writeheader()
#     writer.writerows(data)




