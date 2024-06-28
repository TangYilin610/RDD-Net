import os

import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
from dataloaders.normalize import normalize_image, normalize_image_to_0_1


class OPTIC_dataset(data.Dataset):
    # def __init__(self, root, img_list, label_list, target_size=512, img_normalize=True):
    #     super().__init__()
    #     self.root = root
    #     self.img_list = img_list
    #     self.label_list = label_list
    #     self.len = len(img_list)
    #     self.target_size = (target_size, target_size)
    #     self.img_normalize = img_normalize
    #
    # def __len__(self):
    #     return self.len
    #
    # def __getitem__(self, item):
    #     if self.label_list[item].endswith('tif'):
    #         self.label_list[item] = self.label_list[item].replace('.tif', '-{}.tif'.format(1))
    #     img_file = os.path.join(self.root, self.img_list[item])
    #     label_file = os.path.join(self.root, self.label_list[item])
    #     img = Image.open(img_file)
    #     label = Image.open(label_file).convert('L')
    #
    #     img = img.resize(self.target_size)
    #     label = label.resize(self.target_size, resample=Image.NEAREST)
    #     img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
    #     if self.img_normalize:
    #         # img_npy = normalize_image_to_0_1(img_npy)
    #         for c in range(img_npy.shape[0]):
    #             img_npy[c] = (img_npy[c] - img_npy[c].mean()) / img_npy[c].std()
    #     label_npy = np.array(label)
    #
    #     mask = np.zeros_like(label_npy)
    #     mask[label_npy < 255] = 1
    #     mask[label_npy == 0] = 2
    #     return img_npy, mask[np.newaxis], img_file

    def __init__(self, root, img_list, label_list, target_size=512, img_normalize=True): #target_size=512,391
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        # 读取包含分类标签的CSV文件
        self.label_data = label_list  # 假设label_list为包含分类标签的CSV文件路径
        # self.labels = self.label_data['label'].tolist()  # 假设CSV文件中有一个名为'label'的列存储了分类标签

    def __len__(self):
        return self.len

    def __getitem__(self, item):

        img_file = os.path.join(self.root, self.img_list[item])
        # label = os.path.join(self.root, self.label_list[item])
        label = self.label_data[item]  # 获取对应图像文件的分类标签
        img = Image.open(img_file)

        img = img.resize(self.target_size)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)

        if self.img_normalize:
            for c in range(img_npy.shape[0]):
                img_npy[c] = (img_npy[c] - img_npy[c].mean()) / img_npy[c].std()

        return img_npy, label, img_file



