import numpy as np
import scipy.sparse as sp
import pyscipopt as scip
import datetime
import os
import pyscipopt as scip
import time
import copy
from pyscipopt import quicksum
import random
import torch

def init_scip_params(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648

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
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

class FeaturesCollector(scip.Branchrule):
    def __init__(self):
        self.obs = []

    def branchexeclp(self, allowaddcons):
        if self.model.getNNodes() == 1:
            all_vars = self.model.getVars()

            # candidate_vars, *_ = self.model.getPseudoBranchCands()
            # candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]
            # candidate_mask = np.array(candidate_mask)
            # sorted_indices = np.argsort(candidate_mask)
            # sorted_vars = [candidate_vars[i] for i in sorted_indices]

            var_features, edge_features, cons_features, _ = self.model.getBipartiteGraphRepresentation()
            # var_features91, _, _ = self.model.getBranchFeaturesRepresentation(sorted_vars)
            var_features = [feature[0:16] + [feature[18]] for feature in var_features]
            
            indices = [[row[1] for row in edge_features],[row[0] for row in edge_features]]
            values = [row[2] for row in edge_features]
            mean = sum(values) / len(values)
            squared_diffs = [(x - mean) ** 2 for x in values]
            variance = sum(squared_diffs) / len(squared_diffs)
            std = variance ** 0.5 + 1e-6
            normalized_values = [(x - mean) / std for x in values]
            edge_features_dic = {'indices':indices, 'values':normalized_values}

            self.obs = [{'var_features':var_features, 'cons_features':cons_features, 'edge_features':edge_features_dic}]
            print("success collect features")

        result = self.model.executeBranchRule('pscost', allowaddcons)
        # print(result)
        return {'result':result}

def make_obs(instance, seed):

    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    m.setIntParam('timing/clocktype', 2)  # 1: CPU user seconds, 2: wall clock time
    m.readProblem(f'{instance}')
    print(instance)
    init_scip_params(m, seed=seed, heuristics=False, presolving=False)
    branchrule = FeaturesCollector()
    m.includeBranchrule(branchrule=branchrule,
            name="Collect features branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)
    m.setRealParam('limits/time', 360) # 10s collect features

    m.optimize()

    # get features
    # print(branchrule.obs)
    observation0 = branchrule.obs[0]

    m.freeProb()

    return observation0

def normalize_score(score, neighborhood_size):
    l = 0
    r = 100
    while r - l > 1e-8:
        m = (l + r) * 0.5
        tp_score = torch.pow(score, m)
        tp_sum = torch.sum(tp_score).item()
        if tp_sum > neighborhood_size:
            l = m
        else:
            r = m
    return torch.pow(score, l)

def normalize_score2(logit, neighborhood_size):
    l = 0
    r = 1
    while r - l > 1e-8:
        m = (l + r) * 0.5
        tp_logit = torch.mul(logit, m)
        tp_score = torch.sigmoid(tp_logit)
        tp_sum = torch.sum(tp_score).item()
        if tp_sum < neighborhood_size:
            r = m
        else:
            l = m
    tp_logit = torch.mul(logit, l)
    tp_score = torch.sigmoid(tp_logit)
    return tp_score

def logger(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, mode='a') as f:
            print(str, file=f)

class Solution:
    def __init__(self, model, scip_solution, obj_value):
        self.solution = {}
        for v in model.getVars():
            self.solution[v.name] = scip_solution[v]
        self.obj_value = obj_value

    def value(self, var):
        return self.solution[var.name]

def get_LP_relaxation_solution(model):
    LP_relaxation = scip.Model(sourceModel = model, origcopy = True)
    for var in LP_relaxation.getVars():
        LP_relaxation.chgVarType(var, 'C')
    scip_solve_LP_relaxation_config = {'limits/time' : 300}
    return scip_solve(LP_relaxation, scip_config = scip_solve_LP_relaxation_config)



# def scip_solve(model, scip_config = None, incumbent_solution = None, primal_bound = None,
#                 prev_LNS_log = None, get_num_solutions=1):

#     start_time = time.monotonic()

#     # add a constraint using primal bound 
#     if primal_bound is not None:
#         objective_sense = model.getObjectiveSense()
#         if objective_sense == "minimize":
#             model.addCons(model.getObjective() <= primal_bound + 1e-8)
#         else:
#             model.addCons(model.getObjective() >= primal_bound - 1e-8)

#     # set SCIP
#     if scip_config is not None:
#         for param, value in scip_config.items():
#             model.setParam(param, value)
    
#     # start solving
#     model.optimize()

#     status = model.getGap()
#     log_entry = None
#     # if no solution in a LNS iteration, return the copy of the previous log but change the runtime
#     if model.getNSols() == 0:
#         if prev_LNS_log is None:
#             return -1, None
            
#         log_entry = dict()
#         for k, v in prev_LNS_log.items():
#             log_entry[k] = v

#         end_time = time.monotonic()
#         log_entry['iteration_time'] = end_time - start_time
#         log_entry['solving_time'] = model.getSolvingTime()
#         log_entry['run_time'] = prev_LNS_log['run_time'] + log_entry['iteration_time']
#         return status, log_entry

#     # exist solutions
#     sol = model.getBestSol()
#     obj = model.getSolObjVal(sol)
#     Sol = Solution(model, sol, obj)
#     log_entry = {}
#     log_entry['best_primal_sol'] = Sol
#     log_entry['best_primal_scip_sol'] = sol
#     log_entry['primal_bound'] = obj
    
#     var_index_to_value = dict()
#     for v in model.getVars():
#         v_name = v.name
#         v_value = Sol.value(v)
#         var_index_to_value[v_name] = v_value
#     log_entry['var_index_to_value'] = copy.deepcopy(var_index_to_value)

#     if get_num_solutions > 1:
#         var_index_to_values = dict()
#         for v in model.getVars():
#             var_index_to_values[v.name] = []

#         sol_list = model.getSols()
#         obj_list = []
            
#         sol_list.reverse()

#         for sol in sol_list:
#             Sol = Solution(model, sol, obj)
#             obj = model.getSolObjVal(sol)
#             if primal_bound is not None:
#                 objective_sense = model.getObjectiveSense()
#                 if objective_sense == "minimize":
#                     if obj >= primal_bound - 1e-8: continue
#                 else:
#                     if obj <= primal_bound + 1e-8: continue

#             for v in model.getVars():
#                 v_name = v.name
#                 v_value = Sol.value(v)
                
#                 # v_incumbent_value = incumbent_solution.value(v)
#                 v_incumbent_value = incumbent_solution[str(v.name)]

#                 var_index_to_values[v_name].append(0 if round(v_value) == round(v_incumbent_value) else 1)
#             obj_list.append((obj, primal_bound))

#         log_entry['var_index_to_values'] = copy.deepcopy(var_index_to_values)
#         log_entry['primal_bounds'] = copy.deepcopy(obj_list)
#     else:
#         log_entry['var_index_to_values'] = None
#         log_entry['primal_bounds'] = None

#     end_time = time.monotonic()
#     log_entry['iteration_time'] = end_time - start_time
#     log_entry['solving_time'] = model.getSolvingTime()
#     if prev_LNS_log is not None:
#         log_entry['run_time'] = prev_LNS_log['run_time'] + log_entry['iteration_time']
#     else:
#         log_entry['run_time'] = log_entry['iteration_time']

#     return status, log_entry

def scip_solve(model, scip_config = None, incumbent_solution = None, primal_bound = None,
                prev_LNS_log = None, get_num_solutions=1):

    start_time = time.monotonic()

    # add a constraint using primal bound 
    if primal_bound is not None:
        objective_sense = model.getObjectiveSense()
        if objective_sense == "minimize":
            model.addCons(model.getObjective() <= primal_bound + 1e-8)
        else:
            model.addCons(model.getObjective() >= primal_bound - 1e-8)

    # set SCIP
    if scip_config is not None:
        for param, value in scip_config.items():
            model.setParam(param, value)
    
    # start solving
    model.optimize()

    status = model.getGap()
    log_entry = None
    # if no solution in a LNS iteration, return the copy of the previous log but change the runtime
    if model.getNSols() == 0:
        if prev_LNS_log is None:
            return -1, None
            
        log_entry = dict()
        for k, v in prev_LNS_log.items():
            log_entry[k] = v

        end_time = time.monotonic()
        log_entry['iteration_time'] = end_time - start_time
        log_entry['solving_time'] = model.getSolvingTime()
        log_entry['run_time'] = prev_LNS_log['run_time'] + log_entry['iteration_time']
        return status, log_entry

    # exist solutions
    sol = model.getBestSol()
    obj = model.getSolObjVal(sol)
    Sol = Solution(model, sol, obj)
    log_entry = {}
    log_entry['best_primal_sol'] = Sol
    log_entry['best_primal_scip_sol'] = sol
    log_entry['primal_bound'] = obj
    
    var_index_to_value = dict()
    for v in model.getVars():
        v_name = v.name
        v_value = Sol.value(v)
        var_index_to_value[v_name] = v_value
    log_entry['var_index_to_value'] = copy.deepcopy(var_index_to_value)

    if get_num_solutions > 1:
        var_index_to_values = dict()
        for v in model.getVars():
            var_index_to_values[v.name] = []

        sol_list = model.getSols()
        obj_list = []
            
        sol_list.reverse()

        for sol in sol_list:
            Sol = Solution(model, sol, obj)
            obj = model.getSolObjVal(sol)
            if primal_bound is not None:
                objective_sense = model.getObjectiveSense()
                if objective_sense == "minimize":
                    if obj >= primal_bound - 1e-8: continue
                else:
                    if obj <= primal_bound + 1e-8: continue

            for v in model.getVars():
                v_name = v.name
                v_value = Sol.value(v)
                v_incumbent_value = incumbent_solution.value(v)
                var_index_to_values[v_name].append(0 if round(v_value) == round(v_incumbent_value) else 1)
            obj_list.append((obj, primal_bound))

        log_entry['var_index_to_values'] = copy.deepcopy(var_index_to_values)
        log_entry['primal_bounds'] = copy.deepcopy(obj_list)
    else:
        log_entry['var_index_to_values'] = None
        log_entry['primal_bounds'] = None

    end_time = time.monotonic()
    log_entry['iteration_time'] = end_time - start_time
    log_entry['solving_time'] = model.getSolvingTime()
    if prev_LNS_log is not None:
        log_entry['run_time'] = prev_LNS_log['run_time'] + log_entry['iteration_time']
    else:
        log_entry['run_time'] = log_entry['iteration_time']

    return status, log_entry

def create_neighborhood_with_LB(model, log, neighborhood_size=20, time_limit=300, 
                                improved = None, get_num_solutions=1):

    local_branching_mip = scip.Model(sourceModel = model, origcopy = True)

    incumbent_solution = log['best_primal_sol']
    variables_equal_one = []
    variables_equal_zero = []
    all_int_variables = [v.name for v in local_branching_mip.getVars() if v.vtype() in ["BINARY", "INTEGER"]]
    for v in local_branching_mip.getVars():
        if v.name in all_int_variables:
            v_value = incumbent_solution.value(v)
            if round(v_value) == 1:
                variables_equal_one.append(v)
            else:
                variables_equal_zero.append(v)
    
    local_branching_mip.addCons(quicksum(v for v in variables_equal_zero) + quicksum( (1-v)  for v in variables_equal_one) <= neighborhood_size)
    scip_solve_local_branching_config = {'limits/time' : time_limit}
    destroy_variables = []
        
    print("Solving MIP for local branching ...")
    status, log_entry = scip_solve(local_branching_mip, scip_config = scip_solve_local_branching_config,
                incumbent_solution = incumbent_solution, primal_bound = log['primal_bound'],
                prev_LNS_log = log, get_num_solutions = get_num_solutions)
    local_branching_solution = log_entry['best_primal_sol']

    LB_LP_relaxation_status, LB_LP_relaxation_log_entry = get_LP_relaxation_solution(local_branching_mip)
    if LB_LP_relaxation_log_entry is None:
        LB_LP_relaxation_solution = local_branching_solution
    else:
        LB_LP_relaxation_solution = LB_LP_relaxation_log_entry['best_primal_sol']
            
    tmp_observation = dict()
    tmp_observation["selected_by_LB"] = []

    for v in local_branching_mip.getVars():
        if v.name in all_int_variables:
            v_value = incumbent_solution.value(v)
            v_LB_value = local_branching_solution.value(v)
             
            if round(v_LB_value) == round(v_value): 
                continue
            
            destroy_variables.append(v.name)                            
            tmp_observation["selected_by_LB"].append((v.name, v_value, v_LB_value))

     
    assert len(destroy_variables) <= neighborhood_size
    info = dict()
    info["LB_primal_solution"] = log_entry["primal_bound"]
    info["LB_gap"] = status
    info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
    if get_num_solutions > 1:
        info["multiple_solutions"] = copy.deepcopy(log_entry['var_index_to_values'])
        info["multiple_primal_bounds"] = copy.deepcopy(log_entry['primal_bounds'])
    
    return destroy_variables, info
    
def create_sub_mip(model, destroy_variables, incumbent_solution):
    sub_mip = scip.Model(sourceModel = model, origcopy = True)
    num_free_variables = 0
    all_variables = sub_mip.getVars()

    if len(destroy_variables) > 0:
        if type(destroy_variables[0]) == type("string"):
            destroy_variables_name = copy.deepcopy(destroy_variables)
        else:
            destroy_variables_name = [v.name for v in model.getVars() if v.getIndex() in destroy_variables]
    else:
        destroy_variables_name = []
    
    variables_equal_one = []
    variables_equal_zero = []

    for v in all_variables:
        if not (v.name in destroy_variables_name):
            if not (v.vtype() in ["BINARY", "INTEGER"]): 
                continue

            fixed_value = incumbent_solution.value(v)
            sub_mip.chgVarLb(v, fixed_value)
            sub_mip.chgVarLbGlobal(v, fixed_value)
            sub_mip.chgVarUb(v, fixed_value)
            sub_mip.chgVarUbGlobal(v, fixed_value)
            
        else:
            assert v.vtype() in ["BINARY", "INTEGER"], "destroy variable %s not binary is instead %s"%(v.name, v.vtype())
            v_value = incumbent_solution.value(v)
            if round(v_value) == 1:
                variables_equal_one.append(v)
            else:
                variables_equal_zero.append(v)
            num_free_variables += 1
    
    # print(f"Num of free variables: {num_free_variables}")
    return sub_mip

def get_perturbed_samples(model, destroy_variables, log, sub_time_limit, 
                            num_of_samples_to_generate, int_var):

    var_name_to_index = dict()
    fixed_variables = []
    for i, var in enumerate(int_var):
        var_name_to_index[var.name] = i
        if not (var.name in destroy_variables): 
            fixed_variables.append(var.name)
    
    primal_bound = log['primal_bound']

    objective_sense = model.getObjectiveSense()
    obj_sense = 1 if objective_sense == "minimize" else -1

    collected_samples = []
    primal_bounds = []
    negative_labels = []
    
    for num_of_replaced_variables in range(5, len(destroy_variables)-1, 5):
        no_negative_sample = 0
        for t in range(90):
            
            perturbed_destroy_variables = random.sample(destroy_variables, len(destroy_variables) - num_of_replaced_variables) + random.sample(fixed_variables, num_of_replaced_variables)

            sub_mip = create_sub_mip(model, perturbed_destroy_variables, log['best_primal_sol'])
            scip_solve_destroy_config = {'limits/time' : sub_time_limit}
            status, log_entry = scip_solve(sub_mip, scip_config = scip_solve_destroy_config,
                                            incumbent_solution = log['best_primal_scip_sol'], primal_bound = log['primal_bound'], 
                                            prev_LNS_log = log)
            
            improvement = abs(primal_bound - log_entry["primal_bound"])
            improved = (obj_sense * (primal_bound - log_entry["primal_bound"]) > 1e-5)
            new_primal_bound = log_entry["primal_bound"]

            if not improved:
                print(f"Found negative samples with {num_of_replaced_variables} replaced, primal bound = {primal_bound}, new primal bound = {new_primal_bound}")
                negative_sample = [0] * len(int_var)
                for var_name in perturbed_destroy_variables:
                    negative_sample[var_name_to_index[var_name]] = 1
                collected_samples.append(negative_sample)
                primal_bounds.append((log_entry["primal_bound"], primal_bound))
                negative_labels.append(improvement)
                no_negative_sample = 0
            else:
                no_negative_sample += 1
                if no_negative_sample >= 10: 
                    print(f"No negative samples for 10 consecutive samples with {num_of_replaced_variables} variables replaced")
                    break
                
            if len(collected_samples) == num_of_samples_to_generate:
                return collected_samples, primal_bounds, negative_labels
    
    return collected_samples, primal_bounds, negative_labels

import torch
import torch.nn.functional as F
def pad_tensor(input, pad_sizes, normalize, pad_value=-1e10):
    """
    This function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input.split(pad_sizes.cpu().numpy().tolist())
    processed = []

    for i in range(len(output)):
        slice = output[i]
        if normalize:
            # Normalize the scores to ensure they fall in the [-1, 1] range
            max_val = torch.max(abs(output[i]))
            print(max_val)
            slice /= max_val
        processed.append(F.pad(slice, (0, max_pad_size-slice.size(0)), 'constant', pad_value))

    output = torch.stack(processed, dim=0)
    
    return output

def multi_hot_encoding(input):
    max_val = torch.max(input, -1, keepdim=True).values - 1.0e-10
    multihot = input >= max_val
    return multihot.float()

def augment_variable_features_with_dynamic_ones(batch, device, experiment, problem, window_size=3):

    nh_size_threshold = dict() # filter out training data below certain neighborhood size threshold

    # to add features of the last $window_size improving solutions in LNS
    # each window contains 1. whether we have the solution 2. incumbent values 3. LB relax values
    dynamic_feature_size = window_size * 3
    dynamic_features = torch.zeros((batch.variable_features.shape[0], window_size * 3), dtype = torch.float32)

    tot_variables = 0
    batch_weight = []
    batch_n_candidates = []

    incumbent_history = batch.info["incumbent_history"]
    LB_relaxation_history = batch.info["LB_relaxation_history"]
    
    for i in range(len(incumbent_history)):

        #pop the incumbent solution
        incumbent_history[i].pop()

        number_of_history_added = 0
        number_of_variables = len(LB_relaxation_history[i][0])

        total_candidates = torch.sum(batch.candidate_scores[tot_variables:tot_variables+number_of_variables])
        batch_n_candidates.append(total_candidates)
        
        if problem in nh_size_threshold and  total_candidates<nh_size_threshold[problem]:
            batch_weight.append(0)
            
        else:
            batch_weight.append(1)

        for j in reversed(range(len(incumbent_history[i]))):
            
            assert number_of_variables == len(incumbent_history[i][j])
            assert number_of_variables == len(LB_relaxation_history[i][j])
            dynamic_features[tot_variables:tot_variables+number_of_variables, number_of_history_added*3]  = torch.FloatTensor([1]*number_of_variables)
            dynamic_features[tot_variables:tot_variables+number_of_variables, number_of_history_added*3+1] = torch.FloatTensor(incumbent_history[i][j])
            if "feat1" in experiment:
                dynamic_features[tot_variables:tot_variables+number_of_variables, number_of_history_added*3+2] = torch.zeros(len(LB_relaxation_history[i][j]))
            else:   
                dynamic_features[tot_variables:tot_variables+number_of_variables, number_of_history_added*3+2] = torch.FloatTensor(LB_relaxation_history[i][j])

            number_of_history_added += 1
            if number_of_history_added == window_size:
                break
        
        tot_variables += number_of_variables

    
    assert tot_variables == batch.variable_features.shape[0]
    dynamic_features = dynamic_features.to(device)

    all_features = torch.hstack((batch.variable_features, dynamic_features))
    batch.variable_features = all_features
    
    batch_weight = torch.tensor(batch_weight)
    
    batch.batch_weight = batch_weight.to(device)
    return batch

import networkx as nx
def get_bipartite_graph_representation(m):

    model = scip.Model(sourceModel=m, origcopy=True)

    bg = nx.DiGraph()
    var_name_to_index = dict()
    for var in model.getVars():
        var_name_to_index[var.name] = var.getIndex()
    
    num_var = model.getNVars()
    num_cons = model.getNConss()

    for i in range(num_var):
        bg.add_node(i)
        bg.nodes[i]['bipartite'] = 0
    for i in range(num_cons):
        bg.add_node(i+num_var)
        bg.nodes[i+num_var]['bipartite'] = 1

    all_constraints = model.getConss()
    for i, cons in enumerate(all_constraints):
        var_in_cons = model.getValsLinear(cons)
        for key, value in var_in_cons.items():
            var_index = var_name_to_index[key]
            bg.add_edge(var_index, i + num_var)


    all_variables = list(model.getVars())
    variables_to_nodes = dict()
    for i, feat_dict in bg.nodes(data = True):
        if i < len(all_variables):
            feat_dict.update({"scip_variable": all_variables[i].name})
            variables_to_nodes.update({all_variables[i].name: i})
        else:
            break
    for u, v in bg.edges():
        assert(bg.nodes[u]['bipartite'] == 0)
        assert(bg.nodes[v]['bipartite'] == 1)
    return bg, variables_to_nodes

def create_neighborhood_with_ML(model, log, neighborhood_size, device, ML_info, 
                                    wind_size=3, feature_set="feat2", greedy=True):

    ML_inference_start_time = time.monotonic()
    
    local_branching_mip = scip.Model(sourceModel=model, origcopy=True)
    incumbent_solution = log['best_primal_sol']
    variables_equal_one = []
    variables_equal_zero = []
        
    all_int_variables = [v.name for v in local_branching_mip.getVars() if v.vtype() in ["BINARY", "INTEGER"]]
    for v in local_branching_mip.getVars():
        if v.name in all_int_variables:
            v_value = incumbent_solution.value(v)
            if round(v_value) == 1:
                variables_equal_one.append(v)
            else:
                variables_equal_zero.append(v)

    local_branching_mip.addCons(quicksum(v for v in variables_equal_zero) + quicksum( (1-v)  for v in variables_equal_one) <= neighborhood_size) 

    int_var = [v for v in model.getVars() if v.vtype() in ["BINARY", "INTEGER"]]
    LB_relaxation_solution = []
    if feature_set == "feat1":
        LB_LP_relaxation_solution = log['best_primal_sol']   
        for var in int_var:
            LB_relaxation_solution.append(0)
            
    else:
        LB_LP_relaxation_status, LB_LP_relaxation_log_entry = get_LP_relaxation_solution(local_branching_mip)
        LB_LP_relaxation_solution = LB_LP_relaxation_log_entry['best_primal_sol']
        for var in int_var:
            LB_relaxation_solution.append(LB_LP_relaxation_solution.value(var))
        
    policy, observation, incumbent_history, _LB_relaxation_history = ML_info
    LB_relaxation_history = copy.deepcopy(_LB_relaxation_history)
    LB_relaxation_history.append(LB_relaxation_solution)

    dynamic_features = torch.zeros((observation["variable_features"].shape[0], wind_size * 3), dtype = torch.float32)
    number_of_history_added = 0
    assert(len(incumbent_history) == len(LB_relaxation_history))
    for i in reversed(range(len(LB_relaxation_history))):
        dynamic_features[:, number_of_history_added*3]  = torch.FloatTensor([1]*len(int_var))
        dynamic_features[:, number_of_history_added*3+1] = torch.FloatTensor(incumbent_history[i])

        if feature_set == "feat2":
            dynamic_features[:, number_of_history_added*3+2] = torch.FloatTensor(LB_relaxation_history[i])
        else:
            dynamic_features[:, number_of_history_added*3+2] = torch.zeros(len(LB_relaxation_history[i]))
                
        number_of_history_added += 1
        if number_of_history_added == wind_size:
            break

    variable_features = torch.hstack((observation["variable_features"], dynamic_features))
    constraint_features = observation["constraint_features"]
    edge_indices = observation["edge_indices"]
    edge_features = observation["edge_features"]

    with torch.no_grad():
        obs = ( constraint_features.to(device),
                edge_indices.to(device),
                edge_features.to(device),
                variable_features.to(device) )
        logits = policy(*obs)
        score = torch.sigmoid(logits)
    

    distribution_destroy_variable = []
    all_int_variables = [v.name for v in int_var]
    for i, v in enumerate(model.getVars()):
        if v.name in all_int_variables:
            v_value = score[i].item()
            v_logit = logits[i].item()
            distribution_destroy_variable.append((v.name, v_value, v_logit))
    distribution_destroy_variable.sort(key = lambda x: x[2])

    num_cand = len(distribution_destroy_variable)
    info = dict()
    info["LB_LP_relaxation_solution"] = LB_LP_relaxation_solution
    destroy_variables = []

    ML_inference_end_time = time.monotonic()
    info["ML_time"] = ML_inference_end_time - ML_inference_start_time
        
    best_primal_bound = None
        
    if not greedy:
        normalized_score = normalize_score(score, neighborhood_size)

        if torch.sum(normalized_score).item() > neighborhood_size * 1.5:
            normalized_score = normalize_score2(logits, neighborhood_size)

        for i, v in enumerate(model.getVars()):
            if v.name in all_int_variables:
                v_value = normalized_score[i].item()
                coin_flip = random.uniform(0, 1)
                if coin_flip <= v_value:
                    destroy_variables.append(v.name)

        return destroy_variables, info

    else:
        return [v_name for v_name, _, __ in distribution_destroy_variable[-min(neighborhood_size, num_cand):]], info