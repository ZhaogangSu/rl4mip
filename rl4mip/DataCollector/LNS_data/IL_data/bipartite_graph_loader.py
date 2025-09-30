# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from . import bipartite_graph_dataset as bgd
import torch_geometric
import random

class BipartiteGraphLoader:
    def __init__(self, db, shuffle=True, first_k=None):
        self.shuffle = shuffle
        dbs = db.split('+')
        if len(dbs) == 1:
            self.data = bgd.BipartiteGraphDataset(db, query_opt=not shuffle, read_only=True, first_k=first_k)
        else:
            self.data = bgd.BipartiteGraphDatasets(dbs, query_opt=not shuffle, first_k=first_k)

    def num_examples(self):
        return self.data.sample_cnt
            
    def load(self, batch_size=32, format="pt_geom"):
        
        if format == "pt_geom":
    
            loader = torch_geometric.loader.DataLoader(self.data, batch_size, shuffle=self.shuffle)
            for ptg in loader:
                yield ptg
            return

        assert format == 'ntx'
        k = self.data.len()
        permutation = random.sample(range(k), k)
        batch = []
        for loc in permutation:
            ptg = self.data.get(loc)
            ntx = ptg.to_networkx()
            batch.append(ntx)
            if len(batch) == batch_size:
                yield batch
                batch = []

