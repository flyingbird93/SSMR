# -*- coding: utf-8

import os
import pandas as pd
import pickle
from torch.utils import data
import numpy as np
import torch


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, refer_file):
        # train
        self.sets = list(pd.read_csv(csv_file, index_col=False, sep=',')['image'])
        self.gt_annotations = list(pd.read_csv(csv_file, index_col=False, sep=',')['score'])
        self.gt_norm = list(pd.read_csv(csv_file, index_col=False, sep=',')['norm_score'])
        self.root_dir = root_dir

        self.refer = pd.read_csv(refer_file, header=None, sep=' ', index_col=False)

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        train_feature_path = self.root_dir + str(self.sets[idx].split('.')[0]) # 1, 16928, 5, 5
        # with open(train_feature_path, 'rb') as f:
        #     train_original_feature = pickle.load(f)
        # img_id = int(self.train_img_id_list[idx])

        # refer_feature_path = self.refer_root + str(self.train_img_id_list[idx])
        with open(train_feature_path, 'rb') as f:
            refer_fusion_feature = pickle.load(f)
        # print(refer_fusion_feature.shape)
        # refer_fusion_feature = np.load(refer_feature_path, allow_pickle=True)
        # refer_fusion_feature = np.expand_dims(refer_fusion_feature, axis=0)

        # refer feature
        # print(self.refer[self.refer[0] == self.train_img_id_list[idx]])
        refer_id_list = self.refer[self.refer[0] == self.sets[idx]].iloc[0, 1:].values
        # refer_anno = torch.Tensor(self.anno[self.anno[0] == self.train_img_id_list[idx]].iloc[0, 1:].values).unsqueeze(0)

        for i in range(4):
            # refer_id_list:
            refer_feature_path = self.root_dir + str(refer_id_list[i].split('.')[0])
            with open(refer_feature_path, 'rb') as f:
                refer_feature = pickle.load(f)
            # print(refer_feature.shape)
            # refer_feature = np.expand_dims(refer_feature, axis=0)
            # feature concate
            refer_fusion_feature = np.concatenate((refer_fusion_feature, refer_feature), axis=0) # 7, 1, 16928, 1, 1

        # refer_fusion_feature = np.squeezerefer_fusion_feature)
        # print(refer_fusion_feature.shape)

        score = self.gt_annotations[idx]
        norm_score = self.gt_norm[idx]

        sample = {'refer_feature': refer_fusion_feature, 'score': score, 'norm_score': norm_score}

        return sample
    

class test_AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, refer_file):
        self.sets = list(pd.read_csv(csv_file, index_col=False, sep=',')['image'])
        # self.gt_annotations = list(pd.read_csv(csv_file, index_col=False, sep=',')['score'])
        self.root_dir = root_dir
        
        self.test_refer = pd.read_csv(refer_file, header=None, sep=' ', index_col=False)

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        test_feature_path = self.root_dir + str(self.sets[idx].split('.')[0]) # 1, 16928, 5, 5

        with open(test_feature_path, 'rb') as f:
            refer_fusion_feature = pickle.load(f)

        # refer feature
        refer_id_list = self.test_refer[self.test_refer[0] == self.sets[idx]].iloc[0, 1:].values
        # refer_anno = torch.Tensor(self.anno[self.anno[0] == self.train_img_id_list[idx]].iloc[0, 1:].values).unsqueeze(0)

        for i in range(4):
            refer_feature_path = self.root_dir + str(refer_id_list[i].split('.')[0])
            with open(refer_feature_path, 'rb') as f:
                refer_feature = pickle.load(f)
            # feature concate
            refer_fusion_feature = np.concatenate((refer_fusion_feature, refer_feature), axis=0) # 7, 1, 16928, 1, 1

        sample = {'refer_feature': refer_fusion_feature,  'name': self.sets[idx]}

        return sample
 
