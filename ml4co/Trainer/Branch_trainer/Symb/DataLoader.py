import torch
import numpy as np
import os
import gzip
import pickle
import random

from torch_geometric.data import Data
from torch import as_tensor
import torch_geometric


def is_valid(X):
    return not (np.isnan(X).any() or np.isinf(X).any())

def normalize_features(features):
    features -= features.min(axis=0, keepdims=True)
    max_val = features.max(axis=0, keepdims=True)
    max_val[max_val == 0] = 1
    features /= max_val
    return features

class BranchDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_num, raw_dir_name, processed_suffix):
        super().__init__()
        self.root, self.data_num = root, data_num

        self.raw_dir = os.path.join(self.root, raw_dir_name)
        self.processed_dir = self.raw_dir + processed_suffix

        assert os.path.exists(self.raw_dir) or os.path.exists(self.processed_dir)

        if data_num > 0:
            self.load()
        else:
            self._data_list = []

    def load(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        info_dict_path = os.path.join(self.processed_dir, "info_dict.pt")

        if os.path.exists(info_dict_path):
            info_dict = torch.load(info_dict_path, weights_only=False)
            file_names = info_dict["file_names"]
            processed_files = info_dict["processed_files"]
        else:
            info_dict = {}
            raw_file_names = os.listdir(self.raw_dir)
            random.shuffle(raw_file_names)
            file_names = [os.path.join(self.raw_dir, raw_file_name) for raw_file_name in raw_file_names]
            file_names = [x for x in file_names if not os.path.isdir(x)]
            processed_files = []
            info_dict.update(processed_files=processed_files, file_names=file_names)

        if self.data_num > len(processed_files):
            for file_name in file_names[len(processed_files):self.data_num]:
                with gzip.open(file_name, 'rb') as f:
                    sample = pickle.load(f)
                processed_file = self.process_sample(sample)
                processed_files.append(processed_file)
            self._data_list = processed_files
            
            torch.save(info_dict, info_dict_path)
        else:
            self._data_list = processed_files[:self.data_num]
    
    def process_sample(self, sample):
        raise NotImplementedError

    @property
    def data(self):
        return self._data_list

    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, idx):
        return self._data_list[idx]


class FeatureDataset(BranchDataset):
    def __init__(self, root, data_num, raw_dir_name="train", processed_suffix="_feature_processed"):
        super().__init__(root, data_num, raw_dir_name, processed_suffix)

    def process_sample(self, sample):
        X = np.array(sample["obss"][0][3]) 
        y = sample["obss"][2]["scores"]
        action_set = np.array(sample["obss"][2]["action_set"])
        y = y[action_set]
        assert -1 not in y
        assert is_valid(X) and is_valid(y)
        y = (y>=y.max())
        X = normalize_features(X)
        data = Data(x=as_tensor(X, dtype=torch.float, device="cpu"), y=as_tensor(y, dtype=torch.bool, device="cpu"))
        return data

class BipartiteNodeData(torch_geometric.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "x_cv_edge_index":
            return torch.tensor(
                [[self.x_constraint.size(0)], [self.x_variable.size(0)]]
            )
        if key == "y_cand_mask":
            return self.x_variable.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(BranchDataset):
    def __init__(self, root, data_num, raw_dir_name="train", processed_suffix="_processed"):
        super().__init__(root, data_num, raw_dir_name, processed_suffix)

    def process_sample(self, sample):
        obss = sample["obss"]

        vars_feature, cons_feature, edge = obss[0][0], obss[0][1], obss[0][2]
        depth = obss[2]["depth"]
        scores = obss[2]["scores"]

        action_set = np.array(obss[2]["action_set"])
        indices = sorted(action_set)
        scores = scores[indices]
        labels = scores >= scores.max()

        scores = normalize_features(scores)

        data = BipartiteNodeData(x_constraint=as_tensor(cons_feature, dtype=torch.float, device="cpu"), x_variable=as_tensor(vars_feature, dtype=torch.float, device="cpu"),
                                x_cv_edge_index=as_tensor(edge['indices'], dtype=torch.long, device="cpu"), x_edge_attr=as_tensor(edge['values'], dtype=torch.float, device="cpu"),
                                y_cand_mask=as_tensor(indices, dtype=torch.long, device="cpu"), y_cand_score=as_tensor(scores, dtype=torch.float, device="cpu"), y_cand_label=as_tensor(labels, dtype=torch.bool, device="cpu"),
                                depth=depth)
        return data


def get_all_dataset(problem, datapath, device, graph=False,
                    train_num=1000, valid_num=400, test_num=10000, 
                    batch_size_train=1000, batch_size_valid=400, batch_size_test=1000, 
                    get_train=True, get_valid=True, get_test=False):

    samples_dir = os.path.join(datapath, 'samples')
    file_dir = os.path.join(samples_dir, problem)
    
    if get_train:
        train_dataset = FeatureDataset(file_dir, train_num) if not graph else GraphDataset(file_dir, train_num)
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size_train, shuffle=True, follow_batch=["y"], generator=torch.Generator(device=device)) if not graph else torch_geometric.loader.DataLoader(train_dataset, batch_size_train, shuffle=True, follow_batch=["y_cand_mask"], generator=torch.Generator(device=device))
    else:
        train_loader = None

    if get_valid:
        valid_dataset = FeatureDataset(file_dir, valid_num, raw_dir_name="valid") if not graph else GraphDataset(file_dir, valid_num, raw_dir_name="valid")
        valid_loader = torch_geometric.loader.DataLoader(valid_dataset, batch_size_valid, shuffle=False, follow_batch=["y"]) if not graph else torch_geometric.loader.DataLoader(valid_dataset, batch_size_valid, shuffle=False, follow_batch=["y_cand_mask"])
    else:
        valid_loader = None

    if get_test:
        test_dataset = FeatureDataset(file_dir, test_num, raw_dir_name="test")
        test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size_test, shuffle=False, follow_batch=["y"])
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader
