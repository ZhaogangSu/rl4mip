import os
from os import path as osp
import pickle
import gzip
import torch
import numpy as np
from time import time as time
import random
import sys
import json

import torch_geometric
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch.multiprocessing import Pool
from torch_geometric.data import Data
from torch import as_tensor

import ml4co.Trainer.Nodeselect_model.DSO.utils.utils as utils




class FeatureDataset_NodeSelect(utils.NodeDataset):
    def __init__(self, root, data_num, raw_dir_name="train", processed_suffix="_feature_processed"):
        self.raw_dir_name = raw_dir_name
        self.root = root
        super().__init__(root, data_num, raw_dir_name, processed_suffix)
        "Node Select"
        
    def process_sample(self, file_name_list):
        "node select data"
        import pandas as pd
        # 获取文件夹下所有CSV文件的文件名
        csv_files = file_name_list
        # 创建一个空列表，用来存储特征数据
        features_list = []
        label_list = []
        # 读取每个CSV文件
        for file_path in csv_files:
            # 读取CSV文件
            data = pd.read_csv(file_path, header=None)
            # 提取前40行作为特征
            features = data.iloc[:40].values  # 40行特征，所有列
            label = data.iloc[40].values
            features = features.squeeze()  # 去掉多余的维度
            label = label.squeeze()  # 去掉多余的维度
            # 将特征添加到列表
            features_list.append(features)
            label_list.append(label)
        # 将特征列表转化为二维的 NumPy array
        features_array = np.array(features_list)
        label_array = np.array(label_list)
        assert utils.is_valid(features_array) and utils.is_valid(label_array)
        # features_array = utils.normalize_features(features_array) # 不进行归一化，效果反而好。
        X = features_array
        y = label_array
        y = (y == 1)
        data = Data(x=as_tensor(X, dtype=torch.float, device="cpu"), y=as_tensor(y, dtype=torch.bool, device="cpu"))
        return data
    


class DsoSymbDataLoader:
    # def __init__(datapath='',device='cpu', instance_type='', dataset_type=None, train_num=1000, valid_num=400, test_num=10000, 
    #                 batch_size_train=1000, batch_size_valid=400, batch_size_test=1000, 
    #                 get_train=True, get_valid=True, get_test=False):
    def __init__(self, datapath=None, device=None, **instance_kwargs):

        self.instance_kwargs = instance_kwargs
        self.datapath = datapath
        self.device = device

    def dataloader(self):
        return self.get_all_dataset(datapath=self.datapath, 
                                    device=self.device,
                                    **self.instance_kwargs)


    def get_all_dataset(self, instance_type, dataset_type=None, train_num=1000, valid_num=400, test_num=10000, 
                        batch_size_train=1000, batch_size_valid=400, batch_size_test=1000, get_train=True, get_valid=True, get_test=False, datapath='',device='cpu'):
        
        file_dir = osp.join(datapath, 'behaviours_svm')
        file_dir = osp.join(file_dir, instance_type)
        
        # print("file_dir\n", file_dir)
        if get_train:
            train_dataset = FeatureDataset_NodeSelect(file_dir, train_num)
            train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size_train, shuffle=True, follow_batch=["y"], generator=torch.Generator(device=device))
        else:
            train_loader = None

        if get_valid:
            valid_dataset = FeatureDataset_NodeSelect(file_dir, valid_num, raw_dir_name="valid")
            valid_loader = torch_geometric.loader.DataLoader(valid_dataset, batch_size_valid, shuffle=False, follow_batch=["y"])
        else:
            valid_loader = None

        if get_test:
            test_dataset = FeatureDataset_NodeSelect(file_dir, test_num, raw_dir_name="test")
            test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size_test, shuffle=False, follow_batch=["y"])
        else:
            test_loader = None

        return train_loader, valid_loader, test_loader
