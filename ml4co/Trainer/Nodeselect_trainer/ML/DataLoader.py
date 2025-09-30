#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:08:53 2022

@author: aglabassi
"""

import torch
import torch_geometric
import numpy as np

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        # 修改代码
        # data = torch.load(self.sample_files[idx])
        data = torch.load(self.sample_files[idx], weights_only = False)
        return data



class GNNDataLoader(torch_geometric.loader.DataLoader):
    def __init__(   self, sample_files, batch_size=1, shuffle=True, follow_batch=['constraint_features_s', 
                                                                                            'constraint_features_t',
                                                                                            'variable_features_s',
                                                                                            'variable_features_t']):
        self.dataset = GraphDataset(sample_files)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch)





class RankNetDataLoader():
    def __init__(self, sample_files):
        self.sample_files = sample_files
    

    def get_data(self, files):
        
        X = []
        y = []
        depths = []

        for file in files:
            f_array = np.loadtxt(file)
            features = f_array[:-1]
            comp_res = f_array[-1]
            if features.shape[0]==40:
                X.append(features)
                y.append(comp_res)
                depths.append(np.array([f_array[18], f_array[-3]]))
        return np.array(X), np.array(y), np.array(depths)
        #return np.array(X, dtype=np.float32), np.array(y, dtype=np.long), np.array(depths)
    
    def dataloader(self):
        X, y, _ = self.get_data(self.sample_files)

        X = torch.from_numpy(X)
        y = torch.from_numpy(y).unsqueeze(1)

        return X, y


class SVMDataLoader():
    def __init__(self, sample_files):
        self.sample_files = sample_files
    

    def get_data(self, files):
        
        X = []
        y = []
        depths = []

        for file in files:
            f_array = np.loadtxt(file)
            features = f_array[:-1]
            comp_res = f_array[-1]
            if features.shape[0]==40:
                X.append(features)
                y.append(comp_res)
                depths.append(np.array([f_array[18], f_array[-3]]))
        return np.array(X), np.array(y), np.array(depths)
        #return np.array(X, dtype=np.float32), np.array(y, dtype=np.long), np.array(depths)
    
    def dataloader(self):
        X, y, depths = self.get_data(self.sample_files)

        return X, y, depths