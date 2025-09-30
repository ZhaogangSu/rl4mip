import torch_geometric
import torch

def GNNDataLoaderA(obs, action, graph_num, device):

    Da = action.device
    data = []
    for i in range(len(obs)):
        term = []
        term.append(torch.cat([obs[i]['variable_features'].to(Da), action[i]], dim=-1).to(device))
        term.append(obs[i]['constraint_features'].to(device))
        term.append(obs[i]['edge_indices'].to(device))
        term.append(obs[i]['edge_features'].to(device))
        data.append(term)

    graph_x = GraphDataset(data)
    
    dataloader = torch_geometric.loader.DataLoader(graph_x, batch_size = graph_num, shuffle=False)

    return dataloader

def GNNDataLoader(obs, graph_num, device):

    data = []
    for i in range(len(obs)):
        term = []
        term.append(obs[i]['variable_features'].to(device))
        term.append(obs[i]['constraint_features'].to(device))
        term.append(obs[i]['edge_indices'].to(device))
        term.append(obs[i]['edge_features'].to(device))
        data.append(term)

    graph_x = GraphDataset(data)
    
    dataloader = torch_geometric.loader.DataLoader(graph_x, batch_size = graph_num, shuffle=False)

    return dataloader

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, graphs):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.graphs = graphs  # 存储多个图示例的列表

    def len(self):
        return len(self.graphs)
    
    def get(self, idx):

        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG = self.graphs[idx]

        variable_features, constraint_features, edge_indices, edge_features = BG        

        graph = BipartiteNodeData(
            constraint_features,
            edge_indices.long(),
            edge_features,
            variable_features,
        )    

        graph.num_nodes = len(constraint_features) + len(variable_features)
        graph.cons_nodes = len(constraint_features)
        graph.vars_nodes = len(variable_features)

        return graph

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features



    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
