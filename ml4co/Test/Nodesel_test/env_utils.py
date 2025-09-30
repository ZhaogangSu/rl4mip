#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:36:43 2022

@author: aglabassi
"""
import os
import re
import sys
import numpy as np
from scipy.stats import gmean
from scipy.stats import gstd
import pyscipopt.scip as sp
import pyscipopt
import torch


import time
import torch_geometric

from pyscipopt import Nodesel
from ml4co.Trainer.Nodeselect_model.ML.model import GNNPolicy, RankNet

from line_profiler import LineProfiler
from joblib import dump, load

import json

import torch.nn.functional as F

# from node_selection.recorders import CompFeaturizerSVM, CompFeaturizer, LPFeatureRecorder
# from node_selection.node_selectors import (CustomNodeSelector,
#                                            OracleNodeSelectorAbdel, 
#                                            OracleNodeSelectorEstimator_SVM,
#                                            OracleNodeSelectorEstimator_GP,
#                                            OracleNodeSelectorEstimator_RankNet,
#                                            OracleNodeSelectorEstimator_Symb,
#                                            OracleNodeSelectorEstimator_Symm,
#                                            OracleNodeSelectorEstimator)
# from learning.utils import normalize_graph

# from ..DataCollector.behaviour_utils import CompFeaturizerSVM, CompFeaturizer, LPFeatureRecorder
# from ..DataCollector.behaviour_utils import (CustomNodeSelector,
#                                            OracleNodeSelectorAbdel, 
#                                            OracleNodeSelectorEstimator_SVM,
#                                            OracleNodeSelectorEstimator_GP,
#                                            OracleNodeSelectorEstimator_RankNet,
#                                            OracleNodeSelectorEstimator_Symb,
#                                            OracleNodeSelectorEstimator_Symm,
#                                            OracleNodeSelectorEstimator)


def normalize_graph(constraint_features, 
                    edge_index,
                    edge_attr,
                    variable_features,
                    bounds,
                    depth,
                    bound_normalizor = 1000):
    #SMART
    obj_norm = torch.max(torch.abs(variable_features[:,2]), axis=0)[0].item()
    var_max_bounds = torch.max(torch.abs(variable_features[:,:2]), axis=1, keepdim=True)[0]  
    
    var_max_bounds.add_(var_max_bounds == 0)
    
    var_normalizor = var_max_bounds[edge_index[0]]
    cons_normalizor = constraint_features[edge_index[1], 0:1]
    normalizor = var_normalizor/(cons_normalizor + (cons_normalizor == 0))
    
    variable_features[:,2].div_(obj_norm)
    variable_features[:,:2].div_(var_max_bounds)
    constraint_features[:,0].div_(constraint_features[:,0] + (constraint_features[:,0] == 0) )
    edge_attr.mul_(normalizor)
    bounds.div_(bound_normalizor)
    return (constraint_features, edge_index, edge_attr, variable_features, bounds, depth)



def distribute(n_instance, n_cpu):
    if n_cpu == 1:
        return [(0, n_instance)]
    
    k = n_instance //( n_cpu -1 )
    r = n_instance % (n_cpu - 1 )
    res = []
    for i in range(n_cpu -1):
        res.append( ((k*i), (k*(i+1))) )
    
    res.append(((n_cpu - 1) *k ,(n_cpu - 1) *k + r ))
    return res

def init_scip_params(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648  # SCIP seed range
    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)
    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)
    # no restart
    model.setIntParam('presolving/maxrestarts', 0)
    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)
    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)
    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)
    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
def init_scip_params_4_node(model, seed=9, time_limit=3600):
    # print("使用Node的参数")
    model.setIntParam('randomization/permutationseed',seed) 
    model.setIntParam('randomization/randomseedshift',seed)
    model.setParam('constraints/linear/upgrade/logicor',0)
    model.setParam('constraints/linear/upgrade/indicator',0)
    model.setParam('constraints/linear/upgrade/knapsack', 0)
    model.setParam('constraints/linear/upgrade/setppc', 0)
    model.setParam('constraints/linear/upgrade/xor', 0)
    model.setParam('constraints/linear/upgrade/varbound', 0)
    model.setRealParam('limits/time', time_limit)
    model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()

def init_scip_params_4_branch(model, seed, time_limit=3600, heuristics=True, presolving=True, separating=True, conflict=True):
    model.setIntParam('display/verblevel', 0)

    seed = seed % 2147483648  # SCIP seed range
    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)
    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)
    # no restart
    model.setIntParam('presolving/maxrestarts', 0)
    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)
    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)
    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)
    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)

    model.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
    # time_limit=3600
    gap_limit=0.0
    model.setRealParam('limits/time', time_limit)
    model.setRealParam('limits/gap', gap_limit)


def init_scip_params_4_cut(model, time_limit):
    ""

def init_scip_params_4_intersection(model, seed, time_limit):
    # print("使用并集参数")
    model.setIntParam('randomization/permutationseed',seed) 
    model.setIntParam('randomization/randomseedshift',seed)

    model.setRealParam('limits/time', time_limit)

def init_scip_params_new(model, seed, time_limit=3600, heuristics=True, presolving=True, separating=True, conflict=True):
    # print("AAAAAAAAAAAAAAA")
    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)
    model.setParam("separating/maxroundsroot", 1)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    model.setRealParam('limits/time', time_limit)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)


def get_nodesels2models(nodesels, instance, problem, nums_instances, normalize, device, model_path, scip_para, time_limit):
    
    res = dict()
    nodesels2nodeselectors = dict()
    
    for nodesel in nodesels:
        
        model = sp.Model()
        model.hideOutput()
        model.readProblem(instance)

        if scip_para == 1:
            # 使用和 node 一样的参数
            seed = 9
            init_scip_params_4_node(model, seed, time_limit)
        elif scip_para == 2:
            # 使用和 branch 一样的参数
            seed = 0
            init_scip_params_4_branch(model, seed, time_limit)
        elif scip_para == 3:
            # 使用和 cut 一样的参数
            init_scip_params_4_cut(model, time_limit)
        elif scip_para == 4:
            # 使用的参数的并集
            seed = 0
            init_scip_params_4_intersection(model, seed, time_limit)
        elif scip_para == 5:
            seed = 0
            init_scip_params_new(model, seed, time_limit)
        else:
            # scip_para 不是有效值，抛出异常
            raise ValueError(f"Invalid value for scip_para: {scip_para}. Expected 1, 2, 3, or 4.")


        
        comp = None
        
        if not re.match('default*', nodesel):
            try:
                comp_policy, sel_policy, other = nodesel.split("_")
            except:
                comp_policy, sel_policy = nodesel.split("_")
                

            if comp_policy == 'gnn':
                comp_featurizer = CompFeaturizer()
                
                feature_normalizor = normalize_graph if normalize else lambda x: x
                
                n_primal = int(other.split('=')[-1])
                       
                comp = OracleNodeSelectorEstimator(problem,
                                                   comp_featurizer,
                                                   device,
                                                   feature_normalizor,
                                                   nums_instances,
                                                   use_trained_gnn=True,
                                                   sel_policy=sel_policy,
                                                   n_primal=n_primal,
                                                   model_path=model_path)
                fr = LPFeatureRecorder(model, device)
                comp.set_LP_feature_recorder(fr)
            elif comp_policy == 'svm':
                comp_featurizer = CompFeaturizerSVM(model)
                n_primal = int(other.split('=')[-1])
                comp = OracleNodeSelectorEstimator_SVM(problem, comp_featurizer, nums_instances, sel_policy=sel_policy, n_primal=n_primal, model_path=model_path)
            
            elif comp_policy == 'gp':
                comp_featurizer = CompFeaturizerSVM(model)
                n_primal = int(other.split('=')[-1])
                comp = OracleNodeSelectorEstimator_GP(problem, comp_featurizer, nums_instances, sel_policy=sel_policy, n_primal=n_primal)
                
            elif comp_policy == 'ranknet':
                comp_featurizer = CompFeaturizerSVM(model)
                n_primal = int(other.split('=')[-1])
                comp = OracleNodeSelectorEstimator_RankNet(problem, comp_featurizer, nums_instances, device, sel_policy=sel_policy, n_primal=n_primal, model_path=model_path)
                
            elif comp_policy == 'symb':
                comp_featurizer = CompFeaturizerSVM(model)
                n_primal = int(other.split('=')[-1])
                # 修改代码
                # nums_instances = other.split('=')[-2]
                comp = OracleNodeSelectorEstimator_Symb(problem, comp_featurizer, nums_instances, sel_policy=sel_policy, n_primal=n_primal, model_path=model_path)
            elif comp_policy == 'symm':
                comp_featurizer = CompFeaturizerSVM(model)
                n_primal = int(other.split('=')[-1])
                # 修改代码
                # nums_instances = other.split('=')[-2]
                comp = OracleNodeSelectorEstimator_Symm(problem, comp_featurizer, nums_instances, sel_policy=sel_policy, n_primal=n_primal, model_path=model_path)
            elif comp_policy == 'expert':
                comp = OracleNodeSelectorAbdel('optimal_plunger', optsol=0,inv_proba=0)
                optsol = model.readSolFile(instance.replace(".lp", ".sol"))
                comp.setOptsol(optsol)

            else:
                comp = CustomNodeSelector(comp_policy=comp_policy, sel_policy=sel_policy)

            model.includeNodesel(comp, nodesel, 'testing',  536870911,  536870911)
        
        else:
            _, nsel_name, priority = nodesel.split("_")
            assert(nsel_name in ['estimate', 'dfs', 'bfs']) #to do add other default methods 
            priority = int(priority)
            model.setNodeselPriority(nsel_name, priority)
            

            
        
        res[nodesel] = model
        nodesels2nodeselectors[nodesel] = comp
        
        
        
            
    return res, nodesels2nodeselectors



def get_record_file(now_time, problem, nodesel, instance, size):
    save_dir = os.path.join(os.path.abspath(''),  f'test_node/efficiency_data/{problem}/{size}/{nodesel}/{now_time}/')
    
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        ""
        
    instance = str(instance).split('/')[-1]
    file = os.path.join(save_dir, instance.replace('.lp', '.csv'))
    return file




def get_optimal_data_file(now_time, problem, nodesel, instance, size):
    save_dir = os.path.join(os.path.abspath(''),  f'test_node/optimal_data/{problem}/{size}/{nodesel}/{now_time}/')
    
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        ""
        
    instance = str(instance).split('/')[-1]
    file = os.path.join(save_dir, instance.replace('.lp', '.txt'))
    return file

    
def get_sol_data_file(now_time, problem, nodesel, instance, size):
    save_dir = os.path.join(os.path.abspath(''),  f'test_node/solution_data/{problem}/{size}/{nodesel}/{now_time}/')
    
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        ""
        
    instance = str(instance).split('/')[-1]
    file = os.path.join(save_dir, instance.replace('.lp', '.txt'))
    return file


def record_stats_instance(now_time, problem, nodesel, model, instance, nodesel_obj, worker_id, size):
    # print("size:", size)
    nnode = model.getNNodes()
    time = model.getSolvingTime()
    pd_integral = model.getPrimalDualIntegral()
    gap = model.getGap()

    # 获取最优解
    variables = model.getVars()
    optimal_solution = [model.getSolVal(model.getBestSol(), var) for var in variables]
    # 获取最优值
    optimal_value = float(model.getObjVal())



    # print("最优值:", type(optimal_solution))
    # print("最优解:", type(optimal_value))
    '''
    if nodesel_obj != None:    
        comp_counter = nodesel_obj.comp_counter
        sel_counter = nodesel_obj.sel_counter
    else:
        comp_counter = sel_counter = -1
    
    
    if re.match('gnn*', nodesel):
        init1_time = nodesel_obj.init_solver_cpu
        init2_time = nodesel_obj.init_cpu_gpu
        fe_time = nodesel_obj.fe_time
        fn_time = nodesel_obj.fn_time
        inf_counter = nodesel_obj.inf_counter
        
    else:
        init1_time, init2_time, fe_time, fn_time, inference_time, inf_counter = -1, -1, -1, -1, -1, -1
    
    
    if re.match('svm*', nodesel) or re.match('gp*', nodesel) or re.match('expert*', nodesel) or re.match('ranknet*', nodesel) or re.match('symb*', nodesel):
        inf_counter = nodesel_obj.inf_counter
        inference_time = np.array(nodesel_obj.inference_time).mean()  
    '''
    
    nodesel_substring = nodesel.split('_')[0]

    file = get_record_file(now_time, problem, nodesel_substring, instance, size)

    last_three_dirs = os.sep.join(instance.split(os.sep)[-3:])
    print("Process: ", worker_id, " ", last_three_dirs, ": ", np.array([time, nnode, pd_integral, gap]))

    # np.savetxt(file, np.array([nnode, time, comp_counter, sel_counter, init1_time, init2_time, fe_time, fn_time, inference_time, inf_counter, pd_integral, gap]), delimiter=',')
    np.savetxt(file, np.array([time, nnode, pd_integral, gap]), delimiter=',')

    file_optimal_data = get_optimal_data_file(now_time, problem, nodesel_substring, instance, size) 

    from pathlib import Path
    p = Path(file_optimal_data)
    p.parent.mkdir(parents=True, exist_ok=True)   # 确保目录存在
    with p.open("w", encoding="utf-8") as f:
        f.write("optimal_solution:\n" + "".join(map(str, optimal_solution)) + "\n")
        f.write(f"optimal_value:\n{optimal_value}\n")

    sol_file = get_sol_data_file(now_time, problem, nodesel_substring, instance, size)
    model.writeBestSol(sol_file)
    


def print_infos(problem, nodesel, instance):
    print("------------------------------------------")
    print(f"   |----Solving:  {problem}")
    print(f"   |----Instance: {instance}")
    print(f"   |----Nodesel: {nodesel}")

    

def solve_and_record_default(now_time, problem, instance, verbose):
    default_model = sp.Model()
    default_model.hideOutput()
    default_model.setIntParam('randomization/permutationseed',9) 
    default_model.setIntParam('randomization/randomseedshift',9)
    default_model.readProblem(instance)
    if verbose:
        print_infos(problem, 'default', instance)
    
    default_model.optimize()        
    record_stats_instance(now_time, problem, 'default', default_model, instance, None)

    


#take a list of nodeselectors to evaluate, a list of instance to test on, and the 
#problem type for printing purposes
def record_stats(now_time, nodesels, instances, problem, size, nums_instances, device, normalize, verbose=False, default=True, model_path=None, scip_para=1, worker_id=0, time_limit=3600):

    
    for instance in instances:       
        instance = str(instance)
        
        
        if default and not os.path.isfile(get_record_file(now_time, problem,'default', instance)):
            
            solve_and_record_default(now_time, problem, instance, verbose)
        
        nodesels2models, nodesels2nodeselectors = get_nodesels2models(nodesels, instance, problem, nums_instances, normalize, device, model_path, scip_para, time_limit)
        
        for nodesel in nodesels:  

            model = nodesels2models[nodesel]
            nodeselector = nodesels2nodeselectors[nodesel]
                
           #test nodesels
            nodesel_substring = nodesel.split('_')[0]
            if os.path.isfile(get_record_file(now_time, problem, nodesel_substring, instance, size)): #no need to resolve 
                continue

            if verbose:
                print_infos(problem, nodesel, instance)
            # 在求解一些问题的时候出现：run_test4node.sh: line 3: 3075869 Segmentation fault      (core dumped) python main_test4node.py setcover small gnn 4
            model.optimize()

            # del model
            import gc
            gc.collect()
            # 清理显存
            torch.cuda.empty_cache()

            # print("default is True?", default)
            record_stats_instance(now_time, problem, nodesel, model, instance, nodeselector, worker_id, size)

 
               



def get_mean(now_time, problem, nodesel, instances, stat_type, size):
    res = []
    n = 0
    means = dict()
    # stat_idx = ['nnode', 'time', 'ncomp','nsel', 'init1', 'init2', 'fe', 'fn', 'inf','ninf', 'pd_integral', 'gap'].index(stat_type)
    stat_idx = [ 'time', 'nnode', 'pd_integral', 'gap'].index(stat_type)
    for instance in instances:
        try:
            file = get_record_file(now_time, problem, nodesel, instance, size)
            res.append(np.genfromtxt(file)[stat_idx])
            n += 1
            means[str(instance)] = np.genfromtxt(file)[stat_idx]
        except:
            ''
    
    if stat_type in ['nnode', 'time'] :

        mu = np.exp(np.mean(np.log(np.array(res) + 1 )))

        std = np.exp(np.sqrt(np.mean(  ( np.log(np.array(res)+1) - np.log(mu) )**2 )))
    else:
        mu, std = np.mean(res), np.std(res)

    return mu, n, means, std 


def display_stats_2(now_time, instances_type, size, instances_list, scip_para, method, n_cpu):
    time_mean, num_res, _, time_dev  =  get_mean(now_time, instances_type, method, instances_list, 'time', size)
    nnode_mean, _, _, nnode_dev = get_mean(now_time, instances_type, method, instances_list, 'nnode', size)
    pdi_mean = get_mean(now_time, instances_type, method, instances_list, 'pd_integral', size)[0]
    gap_mean = get_mean(now_time, instances_type, method, instances_list, 'gap', size)[0]

    test_node_efficiency_data = os.path.join(os.path.abspath(''),  f'test_node/efficiency_data/{instances_type}/{size}/{method}/{now_time}/')
    test_node_optimal_data = os.path.join(os.path.abspath(''),  f'test_node/optimal_data/{instances_type}/{size}/{method}/{now_time}/') 
    test_node_sol_data = os.path.join(os.path.abspath(''),  f'test_node/sol_data/{instances_type}/{size}/{method}/{now_time}/')


    # return time_mean, time_dev, nnode_mean, nnode_dev, gap_mean, pdi_mean, num_res
    data_to_append = [
        "==============================================================\n",
        f"now_time: {now_time}\n",
        f"instances: {instances_type}/{size}_test\n",
        f"线程数: {n_cpu}\n",
        f"方法: {method}\n",
        f"参数(1:node,2:branch,3:cut,4:交集,5:默认): {scip_para}\n",
        f"待求解数量: {len(instances_list)}\n",
        f"成功求解数量: {num_res}\n",
        f"求解时间: {time_mean:.2f}±{time_dev:.2f}秒\n",
        f"节点数: {nnode_mean:.2f}±{nnode_dev:.2f}\n",
        f"平均GAP: {gap_mean:.2f}\n",
        f"平均PDI: {pdi_mean:.2f}\n",
        f"求解效率记录保存在: {test_node_efficiency_data}\n",
        # f"最优值与最优解保存在: {test_node_optimal_data}\n",
        f"sol文件保存在: {test_node_sol_data}\n",
        "==============================================================\n"
    ]
    with open('node_output.txt', 'a', encoding='utf-8') as file:
        file.writelines(data_to_append)
    return data_to_append



def display_stats(now_time, problem, data_partition, nums_instances, nodesels, instances, scip_para, min_n='unknown', max_n='unknown', default=False, size=''):
    
    res_info = ""
    
    with open('output.txt','a') as f:
        
        original_stdout = sys.stdout
        sys.stdout = f
        # 修改代码
        nodesels_type = nodesels[0]
        str_index = nodesels_type.find('_')
        substring = nodesels_type[:str_index]
        
        print("======================================================")
        print(f'Statistics on {problem}_{data_partition}_{size}_{substring} for problem size in [{min_n}, {max_n}]')
        print(f'models trained on {nums_instances} instances') 
        print(f'Now_time: {now_time}') 
        print("======================================================")

        res_info += "======================================================"
        res_info += '\n'
        res_info += f'Statistics on {problem}_{data_partition}_{size}_{substring} for problem size in [{min_n}, {max_n}]'
        res_info += '\n'
        res_info += f'models trained on {nums_instances} instances'
        res_info += '\n'
        res_info += f'Now_time: {now_time}'
        res_info += '\n'
        res_info += "======================================================"
        res_info += '\n'

        means_nodes = dict()
        for nodesel in (['default'] if default else []) + nodesels:
            
                
            nnode_mean, n, nnode_means, nnode_dev = get_mean(now_time, problem, nodesel, instances, 'nnode', size)
            time_mean, _, _, time_dev  =  get_mean(now_time, problem, nodesel, instances, 'time', size)
            inf_mean = get_mean(now_time, problem, nodesel, instances, 'inf', size)[0] * 1000
            ncomp_mean = get_mean(now_time, problem, nodesel, instances, 'ncomp', size)[0]
            nsel_mean = get_mean(now_time, problem, nodesel, instances, 'nsel', size)[0]
            pd_mean = get_mean(now_time, problem, nodesel, instances, 'pd_integral', size)[0]
            gap_mean = get_mean(now_time, problem, nodesel, instances, 'gap', size)[0]
            
            
            means_nodes[nodesel] = nnode_means
            
        
            print(f"  {nodesel} ")
            print(f"      Mean over n={n} instances : ")
            print(f"      参数(1:node,2:branch,3:cut,4:交集): {scip_para} : ")
            print(f"        |- B&B Tree Size   :  {nnode_mean:.2f}  ± {nnode_dev:.2f}")

            res_info += f"  {nodesel} "
            res_info += '\n'
            res_info += f"      Mean over n={n} instances : "
            res_info += '\n'
            res_info += f"      参数(1:node,2:branch,3:cut,4:交集): {scip_para} : "
            res_info += '\n'
            res_info += f"        |- B&B Tree Size   :  {nnode_mean:.2f}  ± {nnode_dev:.2f}"


            if re.match('gnn*', nodesel):
                in1_mean = get_mean(now_time, problem, nodesel, instances, 'init1', size)[0]
                in2_mean = get_mean(now_time, problem, nodesel, instances, 'init2', size)[0]
                print(f"        |- Presolving A,b,c Feature Extraction Time :  ")
                print(f"           |---   Init. Solver to CPU:           {in1_mean:.2f}")
                print(f"           |---   Init. CPU to GPU   :           {in2_mean:.2f}")

                res_info += f"        |- Presolving A,b,c Feature Extraction Time :  "
                res_info += '\n'
                res_info += f"           |---   Init. Solver to CPU:           {in1_mean:.2f}"
                res_info += '\n'
                res_info += f"           |---   Init. CPU to GPU   :           {in2_mean:.2f}"
                res_info += '\n'

            print(f"        |- Solving Time    :  {time_mean:.2f}  ± {time_dev:.2f}")
            print(f"        |- PD Integral    :  {pd_mean:.2f} ")
            print(f"        |- PD Gap    :  {gap_mean:.2f} ")

            res_info += f"        |- Solving Time    :  {time_mean:.2f}  ± {time_dev:.2f}"
            res_info += '\n'
            res_info += f"        |- PD Integral    :  {pd_mean:.2f} "
            res_info += '\n'
            res_info += f"        |- PD Gap    :  {gap_mean:.2f} "
            res_info += '\n'

            
            if not re.match('default*', nodesel):
                print(f"        |- Inference Time    :  {inf_mean:.2f} ")

                res_info += f"        |- Inference Time    :  {inf_mean:.2f} "
                res_info += '\n'
            #print(f"    Median number of node created : {np.median(nnodes):.2f}")
            #print(f"    Median solving time           : {np.median(times):.2f}""
        
        
                    
            if re.match('gnn*', nodesel):
                fe_mean = get_mean(now_time, problem, nodesel, instances, 'fe', size)[0]
                fn_mean = get_mean(now_time, problem, nodesel, instances, 'fn', size)[0]
                inf_mean = get_mean(now_time, problem, nodesel, instances, 'inf', size)[0]
                print(f"           |---   On-GPU Feature Updates:        {fe_mean:.2f}")
                print(f"           |---   Feature Normalization:         {fn_mean:.2f}")

                res_info += f"           |---   On-GPU Feature Updates:        {fe_mean:.2f}"
                res_info += '\n'
                res_info += f"           |---   Feature Normalization:         {fn_mean:.2f}"
                res_info += '\n'
                # print(f"           |---   Inference     :                {inf_mean:.2f}")
                

            if not re.match('default*', nodesel):
                print(f"        |- nodecomp calls  :  {ncomp_mean:.0f}")
                res_info += f"        |- nodecomp calls  :  {ncomp_mean:.0f}"
                res_info += '\n'
                if re.match('gnn*', nodesel) or re.match('svm*', nodesel) or re.match('expert*', nodesel) or re.match('ranknet*', nodesel) or re.match('gp*', nodesel) or re.match('symb*', nodesel):
                    inf_counter_mean = get_mean(now_time, problem, nodesel, instances, 'ninf', size)[0]
                    print(f"           |---   inference nodecomp calls:      {inf_counter_mean:.0f}")
                    res_info += f"           |---   inference nodecomp calls:      {inf_counter_mean:.0f}"
                    res_info += '\n'
                print(f"        |- nodesel calls   :  {nsel_mean:.0f}")
                res_info += f"        |- nodesel calls   :  {nsel_mean:.0f}"
                res_info += '\n'
            print("-------------------------------------------------")
            res_info += "-------------------------------------------------"
            res_info += '\n'
        sys.stdout = original_stdout
        print("outputed\n")
        
    return means_nodes, res_info
     
     
    



# ======================================= copy from dso4ns/node_selection/recorders.py =======================================
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
            
        g = (constraint_features,
              edge_indices, 
              edge_features, 
              variable_features, 
              torch.tensor([[lb, -1*ub]], device=self.LP_feature_recorder.device).float(),
              torch.tensor([depth], device=self.LP_feature_recorder.device).float()
              )
        # print("constraint_features:\n", variable_features.shape)
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


#Converts a branch and bound node, aka a sub-LP, to a bipartite var/constraint 
#graph representation
#1LP recorder per problem
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




# ======================================= copy from dso4ns/node_selection/node_selectors.py =======================================


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
        

class OracleNodeSelectorEstimator_RankNet(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, nums_instances, device, sel_policy='', n_primal=2, model_path=None):
        super().__init__(sel_policy=sel_policy)
        
        
        policy = RankNet()
        
        # 修改代码
        # policy.load_state_dict(torch.load(f"/opt/code/dso4ns/checkpoint/policy_{problem}_ranknet_{nums_instances}.pkl", map_location=device)) 
        policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)) #run from main

        policy.to(device)
        
        self.policy = policy
        self.device = device
        

        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
            
        start = time.perf_counter() 
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        decision =  self.policy(torch.tensor(f1, dtype=torch.float, device=self.device), torch.tensor(f2, dtype=torch.float, device=self.device), device=self.device)
        self.inference_time.append(time.perf_counter() - start)
    
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1


class OracleNodeSelectorEstimator_SVM(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, nums_instances, sel_policy='', n_primal=2, model_path=None):
        super().__init__(sel_policy=sel_policy)
        
        self.policy = load(model_path)
        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
        start = time.perf_counter()
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.hstack((f1,f2))
        X = X[np.newaxis, :]        
        
        decision = self.policy.predict(X)[0]
        self.inference_time.append(time.perf_counter() - start)
        
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1

    
class OracleNodeSelectorEstimator_GP(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, nums_instances, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)
        
        self.policy = load(f'./checkpoint/policy_{problem}_gp_{nums_instances}.pkl')
        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
        start = time.perf_counter() 
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.hstack((f1,f2))
        X = X[np.newaxis, :]
        decision = self.policy.predict(X)[0]
        self.inference_time.append(time.perf_counter() - start)
        self.inf_counter += 1
        
        return -1 if decision < 0 else 1

class OracleNodeSelectorEstimator_Symb(CustomNodeSelector):
    '''
     It uses a symbolic regression model to estimate the node comparison.
     The symbolic regression model is trained on the features of the nodes
     and the comparison result.
    '''
    def __init__(self, problem, comp_featurizer, nums_instances, sel_policy='', n_primal=2, model_path=None):
        super().__init__(sel_policy=sel_policy)
        
        self.comp_featurizer = comp_featurizer
        self.problem = problem
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        self.model_path = model_path
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
        start = time.perf_counter() 
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.hstack((f1,f2))

        decision = self.get_test_acc(X)

        # decision = self.policy(X)
        self.inference_time.append(time.perf_counter() - start)
        self.inf_counter += 1
        # print(decision)
        res =  -1 if decision < 0 else 1 
        return res
        
    def get_best_expression_from_json_file(self, instance_type):
        best_expression_file_path = self.model_path
        with open(best_expression_file_path, 'r') as f:
            best_expression_dict = json.load(f)
            return best_expression_dict[instance_type]["expression"] 

    def get_test_acc(self, inputs):
        inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.tensor(inputs)
        expression = self.get_best_expression_from_json_file(self.problem)
        device_cpu = '\'cpu\''
        expression = expression.replace('consts.DEVICE', device_cpu)
        res = F.tanh(eval(expression))
        # print(expression)
        res = res.numpy()
        return res
    
class OracleNodeSelectorEstimator_Symm(CustomNodeSelector):
    '''
     It uses a symbolic regression model to estimate the node comparison.
     The symbolic regression expression is trained on the features of one node for symmetry
     and the comparison result of two nodes.
    '''    
    def __init__(self, problem, comp_featurizer, nums_instances, sel_policy='', n_primal=2, model_path=None):
        super().__init__(sel_policy=sel_policy)
        
        
        self.comp_featurizer = comp_featurizer
        self.inf_counter = 0
        self.problem = problem
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        self.model_path = model_path
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
        start = time.perf_counter() 
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.vstack((f1,f2))

        # decision = 1 / (1 + np.exp(self.policy(X[0]) - self.policy(X[1])))

        decision = 1 / (1 + np.exp(self.get_test_acc(X[0]) - self.get_test_acc(X[1])))

        self.inference_time.append(time.perf_counter() - start)
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1    
    
    def get_best_expression_from_json_file(self, instance_type):
        best_expression_file_path = self.model_path
        with open(best_expression_file_path, 'r') as f:
            best_expression_dict = json.load(f)
            return best_expression_dict[instance_type]["expression"] 

    def get_test_acc(self, inputs):
        inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.tensor(inputs)
        expression = self.get_best_expression_from_json_file(self.problem)
        device_cpu = '\'cpu\''
        expression = expression.replace('consts.DEVICE', device_cpu)
        res = F.tanh(eval(expression))
        # print(expression)
        res = res.numpy()
        return res
    
    
class OracleNodeSelectorEstimator(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, device, feature_normalizor, nums_instances, n_primal=2, use_trained_gnn=True, sel_policy='', model_path=None):
        super().__init__(sel_policy=sel_policy)
        
        
        
        policy = GNNPolicy()
        if use_trained_gnn: 
            # policy.load_state_dict(torch.load(f"./checkpoint/policy_{problem}.pkl", map_location=device)) #run from main
            policy.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("Using randomly initialized gnn")
            
        policy.to(device)
        
        self.policy = policy
        self.comp_featurizer = comp_featurizer
        self.device = device
        self.feature_normalizor = feature_normalizor
        
        self.n_primal = n_primal
        self.best_primal = np.inf #minimization
        self.primal_changes = 0
        
        self.fe_time = 0
        self.fn_time = 0
        self.inference_time = []
        self.inf_counter = 0
        
        self.scores = dict()
        
        
        
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_featurizer.set_LP_feature_recorder(LP_feature_recorder)
        
        self.init_solver_cpu = LP_feature_recorder.init_time
        self.init_cpu_gpu = LP_feature_recorder.init_cpu_gpu_time
        
        self.fe_time = 0
        self.fn_time = 0
        # self.inference_time = []
        self.inf_counter = 0

        
    
    def nodecomp(self, node1,node2):
        
        self.comp_counter += 1        
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
            
            
        #begin inference process
        comp_scores = [-1,-1]
        
        for comp_idx, node in enumerate([node1, node2]):
            n_idx = node.getNumber()
        
            if n_idx in self.scores:
                comp_scores[comp_idx] = self.scores[n_idx]
            else:

                _time, g =  self.comp_featurizer.get_graph_for_inf(self.model, node)
                
                self.fe_time += _time

                
                start = time.time()
                g = self.feature_normalizor(*g)[:-1]
                self.fn_time += (time.time() - start)
                
                start = time.perf_counter()
                score = self.policy.forward_graph(*g).item()
                self.scores[n_idx] = score 
                comp_scores[comp_idx] = score
                self.inference_time.append(time.perf_counter() - start)
                
        self.inf_counter += 1
        
        return -1 if comp_scores[0] > comp_scores[1] else 1