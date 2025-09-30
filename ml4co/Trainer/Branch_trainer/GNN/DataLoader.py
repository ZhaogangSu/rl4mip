import os
import gzip
import pickle
import pathlib
import torch
import torch_geometric
import numpy as np

class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, nb_candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_feats, _ = sample['obss']
        sample_scores = sample_feats['scores']
        sample_action_set = sample_feats['action_set']

        variable_features = sample_observation[0]
        constraint_features = sample_observation[1]
        edge_feats = sample_observation[2]
        edge_indices = edge_feats['indices']
        edge_features = edge_feats['values']
        constraint_features = torch.FloatTensor(np.array(constraint_features, dtype=np.float64))
        edge_indices = torch.LongTensor(np.array(edge_indices, dtype=np.int32))
        edge_features = np.array(edge_features, dtype=np.float32)
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(np.array(variable_features, dtype=np.float64))

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, len(candidates), candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph
    
class GNNdataLoader():
    """"""""
    def __init__(self, problem, datapath):
        samples_dir = os.path.join(datapath, 'samples')
        self.train_files = [str(file) for file in (pathlib.Path(samples_dir)/problem/'train').glob('sample_*.pkl')]
        self.pretrain_files = [f for i, f in enumerate(self.train_files) if i % 10 == 0]
        self.valid_files = [str(file) for file in (pathlib.Path(samples_dir)/problem/'valid').glob('sample_*.pkl')]


    def loadpretraind(self, pretrain_batch_size):
        pretrain_data = GraphDataset(self.pretrain_files)
        pretrain_loader = torch_geometric.loader.DataLoader(pretrain_data, pretrain_batch_size, shuffle=False)
        return pretrain_loader
    
    def loadvalid(self, valid_batch_size):
        valid_data = GraphDataset(self.valid_files)
        valid_loader = torch_geometric.loader.DataLoader(valid_data, valid_batch_size, shuffle=False)
        return valid_loader
    
    def loadepochtrain(self, batch_size, seed=0):
        rng = np.random.RandomState(seed)
        epoch_train_files = rng.choice(self.train_files, int(np.floor(len(self.train_files)/batch_size))*batch_size, replace=True)
        train_data = GraphDataset(epoch_train_files)
        train_loader = torch_geometric.loader.DataLoader(train_data, batch_size, shuffle=True)
        return train_loader


    
