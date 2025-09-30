#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:43:54 2021

@author: abdel
"""
import torch.nn.functional as F

import os
import torch
import sys
import time
import numpy as np
from pyscipopt import Nodesel

from line_profiler import LineProfiler
from joblib import dump, load
import os
import json

import re
import torch_geometric




class CustomNodeSelector(Nodesel):

    def __init__(self, sel_policy='', comp_policy=''):
        self.sel_policy = sel_policy
        self.comp_policy = comp_policy
        self.sel_counter = 0
        self.comp_counter = 0

        
    def nodeselect(self):
        
        self.sel_counter += 1
        policy = self.sel_policy
        
        if policy == 'estimate':
            res = self.estimate_nodeselect()
        elif policy == 'dfs':
            res = self.dfs_nodeselect()
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodeselect()
        elif policy == 'bfs':
            res = self.bfs_nodeselect()
        elif policy == 'random':
            res = self.random_nodeselect()
        else:
            res = {"selnode": self.model.getBestNode()}
            
        return res
    
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        policy = self.comp_policy
        
        if policy == 'estimate':
            res = self.estimate_nodecomp(node1, node2)
        elif policy == 'dfs':
            res = self.dfs_nodecomp(node1, node2)
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodecomp(node1, node2)
        elif policy == 'bfs':
            res = self.bfs_nodecomp(node1, node2)
        elif policy == 'random':
            res = self.random_nodecomp(node1, node2)
        else:
            res = 0
            
        return res
    
    #BFS
    def bfs_nodeselect(self):
        return {'selnode':self.model.getBfsSelNode() }
        
        
        
    #Estimate 
    def estimate_nodeselect(self):
        return {'selnode':self.model.getEstimateSelNode() }
    
    def estimate_nodecomp(self, node1,node2):
        
        #estimate 
        estimate1 = node1.getEstimate()
        estimate2 = node2.getEstimate()
        if (self.model.isInfinity(estimate1) and self.model.isInfinity(estimate2)) or \
            (self.model.isInfinity(-estimate1) and self.model.isInfinity(-estimate2)) or \
            self.model.isEQ(estimate1, estimate2):
                lb1 = node1.getLowerbound()
                lb2 = node2.getLowerbound()
                
                if self.model.isLT(lb1, lb2):
                    return -1
                elif self.model.isGT(lb1, lb2):
                    return 1
                else:
                    ntype1 = node1.getType()
                    ntype2 = node2.getType()
                    CHILD, SIBLING = 3,2
                    
                    if (ntype1 == CHILD and ntype2 != CHILD) or (ntype1 == SIBLING and ntype2 != SIBLING):
                        return -1
                    elif (ntype1 != CHILD and ntype2 == CHILD) or (ntype1 != SIBLING and ntype2 == SIBLING):
                        return 1
                    else:
                        return -self.dfs_nodecomp(node1, node2)
     
        
        elif self.model.isLT(estimate1, estimate2):
            return -1
        else:
            return 1
        
        
        
    # Depth first search        
    def dfs_nodeselect(self):
        
        selnode = self.model.getPrioChild()  #aka best child of current node
        if selnode == None:
            
            selnode = self.model.getPrioSibling() #if current node is a leaf, get 
            # a sibling
            if selnode == None: #if no sibling, just get a leaf
                selnode = self.model.getBestLeaf()
                
        return {"selnode": selnode}
    
    def dfs_nodecomp(self, node1, node2):
        return -node1.getDepth() + node2.getDepth()
    
    
    
    # Breath first search
    def breadthfirst_nodeselect(self):
        
        selnode = self.model.getPrioSibling()
        if selnode == None: #no siblings to be visited (all have been LP-solved), since breath first, 
        #we take the heuristic of taking the best leaves among all leaves
            
            selnode = self.model.getBestLeaf() #DOESTN INCLUDE CURENT NODE CHILD !
            if selnode == None: 
                selnode = self.model.getPrioChild()
        
        return {"selnode": selnode}
    
    def breadthfirst_nodecomp(self, node1, node2): 
        
        d1, d2 = node1.getDepth(), node2.getDepth()
        
        if d1 == d2:
            #choose the first created node
            return node1.getNumber() - node2.getNumber()
        
        #less deep node => better
        return d1 - d2
        
     
     #random
    def random_nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    def random_nodecomp(self, node1,node2):
        return -1 if np.random.rand() < 0.5 else 1

    



class OracleNodeSelectorAbdel(CustomNodeSelector):

    def __init__(self, oracle_type, optsol=0, prune_policy='estimate', inv_proba=0, sel_policy=''):

        super().__init__(sel_policy=sel_policy)
        self.oracle_type = oracle_type
        self.optsol = optsol
        self.prune_policy = prune_policy 
        self.inv_proba = inv_proba
        self.sel_policy = sel_policy
        self.inf_counter  = 0
        self.inference_time = 0
        
    
    def nodecomp(self, node1, node2, return_type=False):
        
        self.comp_counter += 1
        
        if self.oracle_type == "optimal_plunger":            
        
            d1 = self.is_sol_in_domaine(self.optsol, node1)
            d2 = self.is_sol_in_domaine(self.optsol, node2)
            inv = np.random.rand() < self.inv_proba
            
            if d1 and d2:
                res, comp_type = self.dfs_nodecomp(node1, node2), 0
            elif d1:
                res = comp_type = -1
                self.inf_counter += 1
                
            
            elif d2:
                res = comp_type = 1
                self.inf_counter += 1
            
            else:
                res, comp_type = self.estimate_nodecomp(node1, node2), 10              
            
            inv_res = -1 if res == 1 else 1
            res = inv_res if inv else res
            
            return res if not return_type  else  (res, comp_type)
        else:
            raise NotImplementedError

    
    def is_sol_in_domaine(self, sol, node):
        #By partionionning, it is sufficient to only check what variable have
        #been branched and if sol is in [lb, up]_v for v a branched variable
        
        bvars, bbounds, btypes = node.getAncestorBranchings()
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            if btype == 0:#LOWER BOUND
                if sol[bvar] < bbound:
                    return False
            else: #btype==1:#UPPER BOUND
                if sol[bvar] > bbound:
                    return False
        
        return True
            
            
    def setOptsol(self, optsol):
        self.optsol = optsol









# 2014 He He
class CompFeaturizerSVM():
    def __init__(self, model,save_dir=None, instance_name=None):
        self.instance_name = instance_name
        self.save_dir = save_dir
        self.m = model
        
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        f1,f2 = self.get_features(node1), self.get_features(node2)
        
        
        file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.csv")
        file = open(file_path, 'a')
        
        np.savetxt(file, f1, delimiter=',')
        np.savetxt(file, f2, delimiter=',')
        file.write(str(comp_res))
        file.close()
        
        return self
    
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    def get_features(self, node):
        
        model = self.m
        
        f = []
        feat = node.getHeHeaumeEisnerFeatures(model, model.getDepth()+1 )
        
        
        for k in ['vals', 'depth', 'maxdepth' ]:
            if k == 'vals':
                
                for i in range(1,19):
                    try:
                        f.append(feat[k][i])
                    except:
                        f.append(0)
                    
            else:
                f.append(feat[k])

        return f
    
# 原来在 data_type 的
class BipartiteGraphPairData(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation, s is graph0, t is graph1 
    """
    def __init__(self, constraint_features_s=None, edge_indices_s=None, edge_features_s=None, variable_features_s=None, bounds_s=None, depth_s=None, 
                 constraint_features_t=None, edge_indices_t=None, edge_features_t=None, variable_features_t=None,  bounds_t=None, depth_t=None,
                 y=None): 
        
        super().__init__()
        
        self.variable_features_s, self.constraint_features_s, self.edge_index_s, self.edge_attr_s, self.bounds_s, self.depth_s =  (
            variable_features_s, constraint_features_s, edge_indices_s, edge_features_s, bounds_s, depth_s)
        
        self.variable_features_t, self.constraint_features_t, self.edge_index_t, self.edge_attr_t, self.bounds_t, self.depth_t  = (
            variable_features_t, constraint_features_t, edge_indices_t, edge_features_t, bounds_t, depth_t)
        
        self.y = y
        

   
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index_s':
            return torch.tensor([[self.variable_features_s.size(0)], [self.constraint_features_s.size(0)]])
        elif key == 'edge_index_t':
            return torch.tensor([[self.variable_features_t.size(0)], [self.constraint_features_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class CompFeaturizer():
    
    def __init__(self, save_dir=None, instance_name=None):
        self.instance_name = instance_name
        self.save_dir = save_dir
        
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        # LP 特征记录设置
        self.LP_feature_recorder = LP_feature_recorder
        return self
        
    
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        # 比较结果记录
        torch_geometric_data = self.get_torch_geometric_data(model, node1, node2, comp_res)
        file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.pt")
        torch.save(torch_geometric_data, file_path, _use_new_zipfile_serialization=False)
        
        return self
    
    def get_torch_geometric_data(self, model, node1, node2, comp_res=0):
        
        triplet = self.get_triplet_tensors(model, node1, node2, comp_res)
        
        return BipartiteGraphPairData(*triplet[0], *triplet[1], triplet[2])

    def get_graph_for_inf(self,model, node):
        
        gpu_gpu = time.time()
        
        self.LP_feature_recorder.record_sub_milp_graph(model, node)
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g_idx = node.getNumber()
        
        var_attributes, cons_block_idxs = graphidx2graphdata[g_idx]
        
    
        g_data = self._get_graph_data(var_attributes, cons_block_idxs, all_conss_blocks, all_conss_blocks_features)
        
        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        
        lb, ub = node.getLowerbound(), node.getEstimate()
        depth = node.getDepth()
        
        if model.getObjectiveSense() == 'maximize':
            lb,ub = ub,lb
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        g = (constraint_features,
              edge_indices, 
              edge_features, 
              variable_features, 
              torch.tensor([[lb, -1*ub]], device=self.LP_feature_recorder.device).float(),
              torch.tensor([depth], device=self.LP_feature_recorder.device).float()
              )
        gpu_gpu = (time.time() - gpu_gpu)
            
        return gpu_gpu, g

    
    def get_triplet_tensors(self, model, node1, node2, comp_res=0):
        # 
        self.LP_feature_recorder.record_sub_milp_graph(model, node1)
        self.LP_feature_recorder.record_sub_milp_graph(model, node2)
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g0_idx, g1_idx, comp_res = node1.getNumber(), node2.getNumber(), comp_res
        
        var_attributes0, cons_block_idxs0 = graphidx2graphdata[g0_idx]
        var_attributes1, cons_block_idxs1 = graphidx2graphdata[g1_idx]
        
        g_data = self._get_graph_pair_data(var_attributes0, 
                                           var_attributes1, 
                                                         
                                           cons_block_idxs0, 
                                           cons_block_idxs1, 

                                           all_conss_blocks, 
                                           all_conss_blocks_features, 
                                           comp_res)
        
        bounds0 = [node1.getLowerbound(), node1.getEstimate()]
        bounds1 = [node2.getLowerbound(), node2.getEstimate()]
        
        if model.getObjectiveSense() == 'maximize':
            bounds0[1], bounds0[0] = bounds0
            bounds1[1], bounds1[0] = bounds1
            
        return self._to_triplet_tensors(g_data, node1.getDepth(), node2.getDepth(), bounds0, bounds1, self.LP_feature_recorder.device)
    
       
    
    def _get_graph_pair_data(self, var_attributes0, var_attributes1, cons_block_idxs0, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features, comp_res ):
        
        g1 = self._get_graph_data(var_attributes0, cons_block_idxs0, all_conss_blocks, all_conss_blocks_features)
        g2 = self._get_graph_data(var_attributes1, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features)
     
        return list(zip(g1,g2)) + [comp_res]
    
    def _get_graph_data(self, var_attributes, cons_block_idxs, all_conss_blocks, all_conss_blocks_features):
        
        
        adjacency_matrixes = map(all_conss_blocks.__getitem__, cons_block_idxs)
        
        cons_attributes_blocks = map(all_conss_blocks_features.__getitem__, cons_block_idxs)
        
        #TO DO ACCELERATE HSTACK VSTACK
        # adjacency_matrix = torch.hstack(tuple(adjacency_matrixes))
        # cons_attributes = torch.vstack(tuple(cons_attributes_blocks))
        adjacency_matrix = tuple(adjacency_matrixes)[0]
        cons_attributes = tuple(cons_attributes_blocks)[0]
        
        edge_idxs = adjacency_matrix._indices()
        edge_features =  adjacency_matrix._values().unsqueeze(1)
            
        
        return var_attributes, cons_attributes, edge_idxs, edge_features
        
    
    def _to_triplet_tensors(self, g_data, depth0, depth1, bounds0, bounds1, device ):
        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        y = g_data[4]
        lb0, ub0 = bounds0
        lb1, ub1 = bounds1
        
        g1 = (constraint_features[0],
              edge_indices[0], 
              edge_features[0], 
              variable_features[0], 
              torch.tensor([[lb0, -1*ub0]], device=device).float(),
              torch.tensor([depth0], device=device).float()
              )
        g2 = (constraint_features[1], 
              edge_indices[1], 
              edge_features[1], 
              variable_features[1], 
              torch.tensor([[lb1, -1*ub1]], device=device).float(),
              torch.tensor([depth1], device=device).float()
              )
        
        
        
        return (g1,g2,y)


class LPFeatureRecorder():
    
    def __init__(self, model, device):
        
        varrs = model.getVars()
        original_conss = model.getConss()
        
        self.model = model
        
        self.n0 = len(varrs)
        
        self.varrs = varrs
        
        self.original_conss = original_conss
        
        self.recorded = dict()
        self.recorded_light = dict()
        self.all_conss_blocks = []
        self.all_conss_blocks_features = []
        self.obj_adjacency  = None
        
        self.device = device
        
        
        #INITIALISATION OF A,b,c into a graph
        self.init_time = time.time()
        self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])
        root_graph = self.get_root_graph(model, device='cpu')
        self.init_time = (time.time() - self.init_time)
        
        
        self.init_cpu_gpu_time = time.time()
        root_graph.var_attributes = root_graph.var_attributes.to(device)
        for idx, _ in  enumerate(self.all_conss_blocks_features): #1 single loop
            self.all_conss_blocks[idx] = self.all_conss_blocks[idx].to(device)
            self.all_conss_blocks_features[idx] = self.all_conss_blocks_features[idx].to(device)
        
        self.init_cpu_gpu_time = (time.time() - self.init_cpu_gpu_time)
       
        self.recorded[1] = root_graph
        self.recorded_light[1] = (root_graph.var_attributes, root_graph.cons_block_idxs)

   
    def get_graph(self, model, sub_milp):
        
        sub_milp_number = sub_milp.getNumber()
        if sub_milp_number in self.recorded:
            return self.recorded[ sub_milp_number]
        else:
            self.record_sub_milp_graph(model, sub_milp)
            return self.recorded[ sub_milp_number ]
        
    
    def record_sub_milp_graph(self, model, sub_milp):
        
        if sub_milp.getNumber() not in self.recorded:
            
            parent = sub_milp.getParent()
            if parent == None: #Root
                graph = self.get_root_graph(model)
                
            else:
                graph = self.get_graph(model, parent).copy()
                #self._add_conss_to_graph(graph, model, sub_milp.getAddedConss())
                self._change_branched_bounds(graph, sub_milp)
                
            #self._add_scip_obj_cons(model, sub_milp, graph)
            self.recorded[sub_milp.getNumber()] = graph
            self.recorded_light[sub_milp.getNumber()] = (graph.var_attributes, 
                                                         graph.cons_block_idxs)
    
    def get_root_graph(self, model, device=None):
        
        dev = device if device != None else self.device
        
        graph = BipartiteGraphStatic0(self.n0, dev)
        
        self._add_vars_to_graph(graph, model, dev)
        self._add_conss_to_graph(graph, model, self.original_conss, dev)
    
        
        return graph
    
    
    # def _get_obj_adjacency(self, model):
    
    #    if self.obj_adjacency  == None:
    #        var_coeff = { self.var2idx[ str(t[0]) ]:c for (t,c) in model.getObjective().terms.items() if c != 0.0 }
    #        var_idxs = list(var_coeff.keys())
    #        weigths = list(var_coeff.values())
    #        cons_idxs = [0]*len(var_idxs)
           
    #        self.obj_adjacency =  torch.torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, 1), device=self.device)
    #        self.obj_adjacency = torch.hstack((-1*self.obj_adjacency, self.obj_adjacency))
           
    #    return self.obj_adjacency         
       
    
    # def _add_scip_obj_cons(self, model, sub_milp, graph):
    #     adjacency_matrix = self._get_obj_adjacency(model)
    #     cons_feature = torch.tensor([[ sub_milp.getEstimate() ], [ -sub_milp.getLowerbound() ]], device=self.device).float()
    #     graph.cons_block_idxs.append(len(self.all_conss_blocks_features))
    #     self.all_conss_blocks_features.append(cons_feature)
    #     self.all_conss_blocks.append(adjacency_matrix)
  
    
                
    def _add_vars_to_graph(self, graph, model, device=None):
        #add vars
        
        dev = device if device != None else self.device
        
        for idx, var in enumerate(self.varrs):
            graph.var_attributes[idx] = self._get_feature_var(model, var, dev)

    
    def _add_conss_to_graph(self, graph, model, conss, device=None):
        
        dev = device if device != None else self.device

        if len(conss) == 0:
            return

        cons_attributes = torch.zeros(len(conss), graph.d1, device=dev).float()
        var_idxs = []
        cons_idxs = []
        weigths = []
        for cons_idx, cons in enumerate(conss):

            cons_attributes[cons_idx] =  self._get_feature_cons(model, cons, dev)
          
            for var, coeff in model.getValsLinear(cons).items():

                if str(var) in self.var2idx:
                    var_idx = self.var2idx[str(var)]
                elif 't_'+str(var) in self.var2idx:
                    var_idx = self.var2idx['t_' + str(var)]
                else:
                    var_idx = self.var2idx[ '_'.join(str(var).split('_')[1:]) ] 
                    
                var_idxs.append(var_idx)
                cons_idxs.append(cons_idx)
                weigths.append(coeff)


        adjacency_matrix =  torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, len(conss)), device=dev) 
        
        #add idx to graph
        graph.cons_block_idxs.append(len(self.all_conss_blocks_features)) #carreful with parralelization
        #add appropriate structure to self
        self.all_conss_blocks_features.append(cons_attributes)
        self.all_conss_blocks.append(adjacency_matrix)
      

    def _change_branched_bounds(self, graph, sub_milp):
        
        bvars, bbounds, btypes = sub_milp.getParentBranchings()
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            
            if str(bvar) in self.var2idx:
                var_idx = self.var2idx[str(bvar)]
            elif 't_'+str(bvar) in self.var2idx:
                var_idx = self.var2idx['t_' + str(bvar)]
            else:
                var_idx = self.var2idx[ '_'.join(str(bvar).split('_')[1:]) ] 
            
            graph.var_attributes[var_idx, int(btype) ] = bbound
            
        
    
    def _get_feature_cons(self, model, cons, device=None):
        
        dev = device if device != None else self.device
        
        try:
            
            cons_n = str(cons)
            if re.match('flow', cons_n):
                
                rhs = model.getRhs(cons)
                leq = 0
                eq = 1
                geq = 0
            elif re.match('arc', cons_n):
                rhs = 0
                leq = eq =  1
                geq = 0
                
            else:
                rhs = model.getRhs(cons)
                leq = eq = 1
                geq = 0
        except:
            'logicor no repr'
            rhs = 0
            leq = eq = 1
            geq = 0
        
        
        return torch.tensor([ rhs, leq, eq, geq ], device=dev).float()

    def _get_feature_var(self, model, var, device=None):
        
        dev = device if device != None else self.device
        
        lb, ub = var.getLbOriginal(), var.getUbOriginal()
        
        if lb <= - 0.999e+20:
            lb = -300
        if ub >= 0.999e+20:
            ub = 300
            
        objective_coeff = model.getObjective()[var]
        
        binary, integer, continuous = self._one_hot_type(var)
    
        
        return torch.tensor([ lb, ub, objective_coeff, binary, integer, continuous ], device=dev).float()
    
    
    def _one_hot_type(self, var):
        vtype = var.vtype()
        binary, integer, continuous = 0,0,0
        
        if vtype == 'BINARY':
            binary = 1
        elif vtype == 'INTEGER':
            integer = 1
        elif vtype == 'CONTINUOUS':
            continuous = 1
            
        return binary, integer,  continuous

class BipartiteGraphStatic0():
    
    #Defines the structure of the problem solved. Invariant toward problems
    def __init__(self, n0, device, d0=6, d1=4, allocate=True):
        
        self.n0, self.d0, self.d1 = n0, d0, d1
        self.device = device
        
        if allocate:
            self.var_attributes = torch.zeros(n0,d0, device=self.device)
            self.cons_block_idxs = []
        else:
            self.var_attributes = None
            self.cons_block_idxs = None
    
    
    def copy(self):
        
        copy = BipartiteGraphStatic0(self.n0, self.device, allocate=False)
        
        copy.var_attributes = self.var_attributes.clone()
        copy.cons_block_idxs = self.cons_block_idxs #no scip bonds
        
        return copy
