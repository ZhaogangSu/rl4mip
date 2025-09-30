import os
from re import M
import sys
import re
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import multiprocessing as mp
import torch_geometric
import random
import multiprocessing
from functools import partial
import time
import pyscipopt as scip
import torch
import copy
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import cloudpickle as pickle
from functools import cmp_to_key

from ml4co.Trainer.LNS_model.CL_model.gnn_policy import GNNPolicy
from ml4co.DataCollector.LNS_data.CL_data.utils import logger, init_scip_params, make_obs
from ml4co.DataCollector.LNS_data.CL_data.utils import scip_solve, create_neighborhood_with_ML, create_sub_mip
from ml4co.DataCollector.LNS_data.IL_data.utils import get_feat
from ml4co.Trainer.LNS_model.IL_model.gcn_pre import BipartiteNodeData, GNNPolicy_pre

from ml4co.Trainer.LNS_model.GBDT_model.graphcnn import GNNPolicy as GBDT_Policy
from ml4co.Trainer.LNS_model.GBDT_model.gbdt_regressor import GradientBoostingRegressor
from ml4co.Trainer.LNS_trainer.GBDT.generate_blocks import cmp, pair, random_generate_blocks, cross_generate_blocks
from ml4co.Trainer.LNS_trainer.GBDT.cross import cross
from ml4co.Trainer.LNS_trainer.GBDT.gurobi_solver import Gurobi_solver
from ml4co.Test.LNS_test.GNNTest import GNNPolicyTester

def optimize(pair_path: str,
             pickle_path: str,
             graph_model_path: str,
             gbdt_model_path: str,
             fix : float,
             set_time : int,
             rate : float,
             device: str,
             results_loc: str,
             instance_id: str,
             problem: str):
    
    begin_time = time.time()

    with open(pickle_path, "rb") as f:
        matrix = pickle.load(f)

    with open(pair_path, "rb") as f:
        pairs = pickle.load(f)

    policy = GBDT_Policy().to(device)
    # policy.load_state_dict(torch.load(model_path, policy.state_dict()))
    

    try:
        policy.load_state_dict(torch.load(graph_model_path, map_location=device))
        logger(f"Checkpoint {graph_model_path} loaded successfully.", results_loc)
    except Exception as e:
        print(f"Checkpoint {graph_model_path} not found or failed to load, bailing out: {e}")
        logger(f"Checkpoint {graph_model_path} not found or failed to load, bailing out: {e}", results_loc)
        sys.exit(1)


    logits = policy(
        torch.FloatTensor(pairs[1]).to(device),
        torch.LongTensor(pairs[2]).to(device),
        torch.FloatTensor(pairs[3]).to(device),
        torch.FloatTensor(pairs[0]).to(device),
    )
 
    data = logits.tolist()

    obj_type = matrix[0]
    n = matrix[1]
    m = matrix[2]
    k = matrix[3]
    site = matrix[4]
    value = matrix[5]
    constraint = matrix[6]
    constraint_type = matrix[7]
    coefficient = matrix[8]
    lower_bound = matrix[9]
    upper_bound = matrix[10]
    value_type = matrix[11]

    # with open(gbdt_model_path, "rb") as f:

    with open(gbdt_model_path, "rb") as f:
        GBDT = pickle.load(f)[0]

    b_time = time.time()
    predict = GBDT.predict(np.array(data)) # 预测初始解
    loss =  GBDT.calc(np.array(data))
    e_time = time.time()
    print("predict time: ", e_time - b_time)
    values = []
    for i in range(n):
        values.append(pair())
        values[i].site= i
        values[i].loss = loss[i]
    
    random.shuffle(values)
    values.sort(key = cmp_to_key(cmp))  

    set_rate = 1
    
    for turn in range(10):
        
        obj = (int)(n * (1 - set_rate * rate))

        solution = []
        color = []
        for i in range(n):
            solution.append(0) # 解
            color.append(0) # 颜色

        for i in range(n):
            now_site = values[i].site
            if(i < obj):
                if(predict[now_site] < 0.5):
                    solution[now_site] = 0
                else:
                    solution[now_site] = 1
            else:
                color[now_site] = 1

        for i in range(m):
            constr = 0
            flag = 0
            for j in range(k[i]):
                if(color[site[i][j]] == 1):
                    flag = 1
                else:
                    constr += solution[site[i][j]] * value[i][j]

            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    for j in range(k[i]):
                        if(color[site[i][j]] == 0):
                            color[site[i][j]] = 1
                            obj -= 1
                            constr -= solution[site[i][j]] * value[i][j]
            else:
                if(constr + flag < constraint[i]):
                    for j in range(k[i]):
                        if(color[site[i][j]] == 0):
                            color[site[i][j]] = 1
                            obj -= 1
                            break
        print(obj / n)
        if(obj / n + rate >= 1):
            break
        else:
            set_rate -= 0.1

    log_entry = dict()

    ansTime = []
    ansVal = []

    nowX, nowVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, (0.5 * set_time), obj_type, lower_bound, upper_bound, value_type, solution, color) 
    #print("nowX", nowX)
    #print("nowVal", nowVal)
    ansTime.append(time.time() - begin_time)
    ansVal.append(nowVal)

    log_entry['iteration_time'] = time.time() - begin_time
    log_entry['run_time'] = time.time() - begin_time
    log_entry['primal_bound'] = nowVal
    log_entry['best_primal_scip_sol'] = nowX
    LNS_log = [log_entry]

    logger(f"Problem {instance_id}: start_time_points: {log_entry['run_time']}", logfile=results_loc)
    logger(f"Problem {instance_id}: start_primal_bounds: {LNS_log[0]['primal_bound']}", logfile=results_loc)

    logger("Start LNS iteration ...", results_loc)
    logger(f"Solving time limit: {set_time}", results_loc)

    random_flag = 0
    count = 0

    while(time.time() - begin_time < set_time):

        start_time = time.time()
        count += 1
        
        logger(f"Problem {instance_id}: start solve step: {count}", logfile = results_loc)
        
        if(random_flag == 1):
            turnX = []
            for i in range(n):
                turnX.append(0)
            if(obj_type == 'maximize'):
                turnVal = 0
            else:
                turnVal = 1e9
            block_list, score, _ = random_generate_blocks(n, m, k, site, values, loss, fix, rate, predict, nowX)
            neibor_num = len(block_list)
            for turn in range(int(1 / rate)):
                i = 0
                now_loss = 0
                for j in range(3):
                    now_site = random.randint(0, neibor_num - 1)
                    if(score[now_site] > now_loss):
                        now_loss = score[now_site]
                        i = now_site
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(max_time, 0.2 * set_time), obj_type, lower_bound, upper_bound, value_type, nowX, block_list[i])
                if(newVal == -1):
                    continue
                if(obj_type == 'maximize'):
                    if(newVal > turnVal):
                        turnVal = newVal
                        for j in range(n):
                            turnX[j] = newX[j]
                else:
                    if(newVal < turnVal):
                        turnVal = newVal
                        for j in range(n):
                            turnX[j] = newX[j]
            if(obj_type == 'maximize'):
                if(turnVal == 0):
                    continue
            else:
                if(turnVal == 1e9):
                    continue

            for i in range(n):
                nowX[i] = turnX[i]
            nowVal = turnVal
            ansTime.append(time.time() - begin_time)
            ansVal.append(nowVal)
        else:
            turnX = []
            turnVal = []
            block_list, _, _ = cross_generate_blocks(n, loss, rate, predict, nowX, GBDT, data)
            for i in range(4):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(max_time, 0.2 * set_time), obj_type, lower_bound, upper_bound, value_type, nowX, block_list[i])
                if(newVal == -1):
                    continue
                turnX.append(newX)
                turnVal.append(newVal)
            
            #cross
            if(len(turnX) == 4):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[0], block_list[0], turnX[1], block_list[1], min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newVal != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)

                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[2], block_list[2], turnX[3], block_list[3],  min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newVal != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)
            if(len(turnX) == 6):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break

                block_list.append(np.zeros(n, int))
                for i in range(n):
                    if(block_list[0][i] == 1 or block_list[1][i] == 1):
                        block_list[4][i] = 1
                block_list.append(np.zeros(n, int))
                for i in range(n):
                    if(block_list[2][i] == 1 or block_list[3][i] == 1):
                        block_list[5][i] = 1
                
                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[4], block_list[4], turnX[5], block_list[5], min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newX != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)
            
            for i in range(len(turnVal)):
                if(obj_type == 'maximize'):
                    if(turnVal[i] > nowVal):
                        nowVal = turnVal[i]
                        for j in range(n):
                            nowX[j] = turnX[i][j]
                else:
                    if(turnVal[i] < nowVal):
                        nowVal = turnVal[i]
                        for j in range(n):
                            nowX[j] = turnX[i][j]
            
            ansTime.append(time.time() - begin_time)
            ansVal.append(nowVal)
            # print(rate, "IS")
            print("time:", time.time() - begin_time, "nowVal:", nowVal)
        
        # logger(f"Step: {count} ,Time: {time.time() - begin_time}, Best obj: {nowVal}", results_loc)
        
        log_entry['iteration_time'] = time.time() - start_time
        log_entry['run_time'] = time.time() - begin_time
        log_entry['primal_bound'] = nowVal
        log_entry['best_primal_scip_sol'] = nowX
        LNS_log.append(log_entry)

        logger(f"Problem {instance_id}: Finished LNS step {count}: obj_val = {log_entry['primal_bound']} in iteration time {log_entry['iteration_time']}", logfile = results_loc)

    if ansTime and ansVal:
        logger(
            f"Problem {instance_id}: initial solution obj = {ansVal[0]}, "
            f"found in time {ansTime[0]:.2f}s",
            logfile=results_loc
        )

    # 记录最优解信息（从所有时间点中找最优）
    
    if ansVal:
        if problem == "cauctions" or problem == "indset":
            best_idx = np.argmax(ansVal)  # 假设是最小化问题
        elif problem == "setcover" or problem == "mvc":
            best_idx = np.argmin(ansVal)  # 假设是最大化问题
        else:
            raise ValueError(f"Unknown problem type: {problem}")
        logger(
            f"Problem {instance_id}: best primal bound {ansVal[best_idx]}, "
            f"found at time {ansTime[best_idx]:.2f}s, "
            f"total time {ansTime[-1]:.2f}s",
            logfile=results_loc
        )  

    
    return LNS_log, ansTime, ansVal

class Solution:
    def __init__(self, model, scip_solution, obj_value):
        self.solution = {}
        for v in model.getVars():
            self.solution[v.name] = scip_solution[v]
        self.obj_value = obj_value

    def value(self, var):
        return self.solution[var.name]
    
def diving_trial(seed, instance, preds, ratio, prev_LNS_log, start_time, seed_id, primal_bound=None, incumbent_solution=None):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if start_time is None:
        start_time = time.monotonic()

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = scip.Model()
    
    model.readProblem(f'{instance}')

    model.setRealParam("limits/time", 30)
    vars = model.getVars()

    fix_by_confident_sampling(model, vars, preds, ratio, seed, seed_id)

    model.optimize()


    status = model.getGap()
    log_entry = None

    # 如果没有解，返回前一个log的copy，但更新时间
    get_num_solutions = model.getNSols()

    if get_num_solutions == 0:
        if prev_LNS_log is None:
            return -1, None

        log_entry = dict()
        for k, v in prev_LNS_log.items():
            log_entry[k] = v

        end_time = time.monotonic()
        log_entry['primal_bound'] = None
        log_entry['iteration_time'] = end_time - start_time
        log_entry['solving_time'] = model.getSolvingTime()
        log_entry['run_time'] = prev_LNS_log.get('run_time', 0) + log_entry['iteration_time']
        return status, log_entry

    # 有解的情况
    sol = model.getBestSol()
    obj = model.getSolObjVal(sol)
    Sol = Solution(model, sol, obj)
    log_entry = {}
    log_entry['best_primal_sol'] = Sol
    # log_entry['best_primal_scip_sol'] = sol
    log_entry['primal_bound'] = obj

    var_index_to_value = {}
    for v in model.getVars():
        v_name = v.name
        v_value = Sol.value(v)
        var_index_to_value[v_name] = v_value
    log_entry['var_index_to_value'] = copy.deepcopy(var_index_to_value)

    if get_num_solutions > 1:
        var_index_to_values = {}
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
                    if obj <= primal_bound - 1e-8:
                        continue
                else:
                    if obj >= primal_bound + 1e-8:
                        continue

            for v in model.getVars():
                v_name = v.name
                v_value = Sol.value(v)
                v_incumbent_value = incumbent_solution.value(v) if incumbent_solution is not None else 0
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
        log_entry['run_time'] = prev_LNS_log.get('run_time', 0) + log_entry['iteration_time']
    else:
        log_entry['run_time'] = log_entry['iteration_time']

    return status, log_entry
# def fix_by_uncertainty_sampling(model, vars, preds, ratio):
    
#     preds_np = preds.detach().cpu().numpy()
#     uncertainty = np.minimum(preds_np, 1 - preds_np)
#     prob_dist = uncertainty / (uncertainty.sum() + 1e-8)
#     num_to_fix = int(len(vars) * ratio)
#     selected_indices = np.random.choice(len(vars), size=num_to_fix, replace=False, p=prob_dist)
#     for i in selected_indices:
#         fix_val = 0.0 if preds_np[i] < 0.5 else 1.0
#         model.fixVar(vars[i], fix_val)

def fix_by_confident_sampling(model, vars, preds, ratio, seed, branch_id=0):
    """
    基于高置信度的概率采样策略，在保持稳定性的同时提供一定探索性。
    
    参数：
        model: SCIP 模型
        vars: 模型变量列表
        preds: torch.Tensor，模型预测值 ∈ [0, 1]
        ratio: float，固定比例（如 0.3）
        branch_id: 当前并行分支编号，用于采样多样性
        seed_base: 基础随机种子
    """
    preds_np = preds.detach().cpu().numpy()
    confidence = np.abs(preds_np - 0.5)  # 越接近0或1，置信度越高
    prob_dist = confidence / (confidence.sum() + 1e-8)  # 归一化为概率分布

    np.random.seed(seed + branch_id)  # 保证每个分支采样不同
    num_to_fix = int(len(vars) * ratio)
    selected_indices = np.random.choice(
        len(vars), size=num_to_fix, replace=False, p=prob_dist
    )

    for i in selected_indices:
        fix_val = 0.0 if preds_np[i] < 0.5 else 1.0
        model.fixVar(vars[i], fix_val)

class CLArgs:
    def __init__(self):
        self.seed = 0

        # 测试参数

        self.init_time_limit = 10
        self.neighborhood_size = 100  # 对应 --neighborhood-size
        self.num_solve_steps = 3000  # 对应 --num-solve-steps
        self.sub_time_limit = 120

        self.gnn_type = "gat"
        self.feature_set = "feat2"
        self.greedy = True
        self.wind_size = 3
        self.adaptive = 1.02

        # 预训练测试参数

class NDPolicyTestEnv:
    
    def __init__(self, sampling_rate):
        
        self.seed = 0
        self.sampling_rate = sampling_rate

    def pre_solve(self, bg_data, instance, device, args): 

        start_time = time.monotonic()
        
        A, v_map, v_nodes, c_nodes, b_vars = bg_data

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features =A._values().unsqueeze(1)
        edge_features=torch.ones(edge_features.shape)
        
        constraint_features[torch.isnan(constraint_features.cpu())] = 1
    

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            torch.FloatTensor(variable_features.cpu()),
        )

        pre_policy = GNNPolicy_pre()
        try:
            ckpt = torch.load(args.pre_model_path, map_location=device, weights_only=False)
            try_again = False
        except Exception as e:
            print("Checkpoint " + args.pre_model_path + " not found, bailing out: " + str(e))
            sys.exit(1)

        pre_policy.load_state_dict(ckpt)
        pre_policy = pre_policy.to(device)
        pre_policy.eval()

        with torch.no_grad():
            BD = pre_policy(
                graph.constraint_features.to(device),
                graph.edge_index.to(device),
                graph.edge_attr.to(device),
                graph.variable_features.to(device),
            ).sigmoid().squeeze().to("cpu")

        prev_LNS_log = {}


        diving_trial_fn = partial(
            diving_trial,
            args.seed,
            instance,
            BD,
            self.sampling_rate,
            prev_LNS_log,
            start_time,
        )
        
        mp.set_start_method('spawn', force=True)

        from multiprocessing.pool import ThreadPool

        with ThreadPool(processes=4) as pool:  # 使用线程池替代进程池
            seeds = list(range(4))
            results = pool.map(diving_trial_fn, seeds)
        # with Pool(processes=4) as pool:
        #     seeds = list(range(4))
        #     results = pool.map(diving_trial_fn, seeds)

        # 过滤掉无解的情况（例如 primal_bound 为 None）
        filtered = [r for r in results if r[1]['primal_bound'] is not None]

        # 如果没有任何解，防止空列表报错
        if not filtered:
            best_result = None
        else:
            best_result = max(filtered, key=lambda x: x[1]['primal_bound'])

        # 展示最优的
        if best_result is not None:
            best_status, best_log = best_result
            print(f"Best bound: {best_log['primal_bound']}")
        else:
            print("No feasible solution found in any trial.")

        return best_status, best_log

# def _worker_process(self, episode_instance, instances_dir, logfile, args, NDPolicy):
#     """
#     子进程中处理单个实例的函数。
#     episode_instance: (episode_index, instance_filename)
#     """
#     episode, inst_fname = episode_instance
#     inst_path = os.path.join(instances_dir, inst_fname)

#     logger(f"Worker started for episode {episode}", logfile=logfile)
#     logger(f"Processing instance {inst_fname}", logfile=logfile)

#     # with open(logfile, 'a') as f:
#     #     f.write(f"[Episode {episode}] Processing {inst_fname}\n")
#     # 送订单并测试
#     orders_queue = self.send_orders(episode, inst_path, self.device, logfile, args)
#     results_LNS = self.test_single_instance(orders_queue, NDPolicy)
#     # 返回最终的 primal_bound
#     if results_LNS and isinstance(results_LNS[-1], dict) and 'primal_bound' in results_LNS[-1]:
#         return results_LNS[-1]['primal_bound']
#     else:
#         print(f"[WARNING] Missing 'primal_bound' in result: {results_LNS[-1] if results_LNS else 'Empty results'}")
#         return {'primal_bound': None}  # 或者其他合适的默认值/结构

def _worker_process(self, episode_instance, instances_dir, logfile, args, NDPolicy):
    
    """
    子进程中处理单个实例的函数。
    episode_instance: (episode_index, instance_filename)
    """

    episode, inst_fname = episode_instance
    inst_path = os.path.join(instances_dir, inst_fname)

    logger(f"Worker started for episode {episode}", logfile=logfile)
    logger(f"Processing instance {inst_fname}", logfile=logfile)

    # 送订单并测试
    orders_queue = self.send_orders(episode, inst_path, self.device, logfile, args)
    results_LNS, metric_log = self.test_single_instance(orders_queue, NDPolicy)

    if results_LNS and isinstance(results_LNS[-1], dict) and 'primal_bound' in results_LNS[-1]:
        # 可选：添加更多结构字段
        return {
            "episode": episode,
            "instance": inst_fname,
            "final_primal_bound": results_LNS[-1]['primal_bound'],
            "total_time": metric_log["total_runtime"],
            "num_steps": metric_log["num_steps"],
            "time_points": metric_log["time_points"],
            "primal_bounds": metric_log["primal_bounds"],
            "variable_assignments": metric_log["variable_assignments"],
            "log": metric_log,
        }
    else:
        logger(f"[WARNING] Missing 'primal_bound' in result: {results_LNS[-1] if results_LNS else 'Empty results'}", logfile)
        return {
            "episode": episode,
            "instance": inst_fname,
            "final_primal_bound": None,
            "error": "Missing 'primal_bound'",
            "log": None,
        }


class LNSPolicyTestEnv:
    def __init__(self, problem, data_path, path_save_results, seed=0):

        self.seed = 0
        self.problem = problem
        self.data_path = data_path
        self.seed = seed
        self.path_save_results = path_save_results

    def test_single_instance(self, in_queue, NDPolicy):

        episode, instance, device, seed, results_loc, args = in_queue

        instance_id = instance.split('/')[-1].split(".lp")[0]
        
        start_run = time.monotonic()

        logger("Testing instance: {}".format(instance_id), results_loc)
        logger("Instance path: {}".format(instance), results_loc)
        logger("start_run: {}".format(start_run), results_loc)
        logger("seed: {}".format(seed), results_loc)

        log_entry = None

        if self.method == "lns_CL" or self.method == "lns_IL":
            model = scip.Model()
            model.setIntParam('display/verblevel', 0)
            model.setIntParam('timing/clocktype', 2)  # 1: CPU user seconds, 2: wall clock time
            model.readProblem(f'{instance}')

            logger(
                f"type: start, episode: {episode}, instance: {instance}, seed: {seed}", 
            results_loc)

            # start LNS


            # get features
            observation0 = make_obs(instance, seed)
            constraint_features = torch.FloatTensor(np.array(observation0["cons_features"], dtype=np.float64))
            variable_features = torch.FloatTensor(np.array(observation0["var_features"], dtype=np.float64))
            edge_indices = torch.LongTensor(np.array(observation0["edge_features"]["indices"], dtype=np.int32))
            edge_features = torch.FloatTensor(np.expand_dims(np.array(observation0["edge_features"]["values"], dtype=np.float32), axis=-1))
            
            observation = {"variable_features":variable_features, "constraint_features":constraint_features,
                            "edge_indices":edge_indices, "edge_features":edge_features}

            # use model to LNS
            init_scip_params(model, seed=seed, presolving=True)
            vars = model.getVars()
            int_var = [v for v in vars if v.vtype() in ["BINARY", "INTEGER"]]
            num_int_var = len(int_var)
            objective_sense = model.getObjectiveSense()
            obj_sense = 1 if objective_sense == "minimize" else -1

            logger(f"Processing instance: {instance} ...", results_loc)
            logger(f"Num of integer variables: {len(int_var)}", results_loc)
            if len(observation0["var_features"]) != len(int_var):
                logger('variable features error', results_loc)

            # initial bipartitegraph
            # bg ,variables_to_nodes = get_bipartite_graph_representation(m=model)

            # find initial solution with SCIP in 10s
            scip_solve_init_config = {'limits/solutions' :10000, 'limits/time' : args.init_time_limit}
            
            if self.method == 'lns_CL':
                status, log_entry = scip_solve(model, scip_config = scip_solve_init_config)
            elif self.method == 'lns_IL':
                bg_data = get_feat(instance)
                status, log_entry = NDPolicy.pre_solve(bg_data, instance, device, args)

            logger("Solving MIP for init ...", results_loc)

            if log_entry is None:
                logger(f'{instance} did not find initial solution', results_loc)
            
            logger(f"Problem {instance_id} initial solution obj = {log_entry['primal_bound']}, found in time {log_entry['run_time']}", results_loc)
            LNS_log = [log_entry]
            runtime_used = log_entry['run_time']

            # initialize incumbent_history with the initial solution
            incumbent_solution = []
            incumbent_history = []
            improvement_history = []
            LB_relaxation_history = []

            for var in int_var:        
                incumbent_solution.append(log_entry["var_index_to_value"][var.name])
            incumbent_history.append(incumbent_solution)

            # load ML model
            policy = GNNPolicy(args.gnn_type)
            
            try:
                ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
                logger(f"Checkpoint {args.model_path} loaded successfully.", results_loc)
                try_again = False
            except Exception as e:
                print(f"Checkpoint {args.model_path} not found or failed to load, bailing out: {e}")
                logger(f"Checkpoint {args.model_path} not found or failed to load, bailing out: {e}", results_loc)
                sys.exit(1)

            policy.load_state_dict(ckpt.state_dict())
            policy = policy.to(device)
            logger(f"{args.gnn_type} will run test on {device} device", results_loc)

            # start LNS steps
            neighborhood_size = args.neighborhood_size
            greedy = args.greedy
            logger("Start LNS iteration ...", results_loc)
            logger(f"Solving steps limit: {args.num_solve_steps}", results_loc)
            logger(f"Original Neighborhood size: {neighborhood_size}", results_loc)
            
            # 初始化记录
            time_points = []
            primal_bounds = []
            variable_assignments = []  # 可用于后续 PG 计算或验证可行性

            # 初始解记录
            time_points.append(runtime_used)
            primal_bounds.append(LNS_log[0]['primal_bound'])
            variable_assignments.append(LNS_log[0]["var_index_to_value"])

            print("start_time_points: ", runtime_used)
            print("start_primal_bounds: ", LNS_log[0]['primal_bound'])
            logger(f"Problem {instance_id}: start_time_points: {runtime_used}", logfile=results_loc)
            logger(f"Problem {instance_id}: start_primal_bounds: {LNS_log[0]['primal_bound']}", logfile=results_loc)

            for s in range(args.num_solve_steps):
                
                print(f"Problem {instance_id}: start solve step: ", s)
                logger(f"Problem {instance_id}: start solve step: {s}", logfile = results_loc)

                iteration_start_time = time.monotonic()
                # best_sol = LNS_log[-1]['best_primal_scip_sol']
                            
                best_sol = LNS_log[-1]['best_primal_sol']

                primal_bound = LNS_log[-1]['primal_bound']

                ML_info = (policy, observation, incumbent_history, LB_relaxation_history)

                # use ML model to get destroy variables
                destroy_variables, info_destroy_heuristic = create_neighborhood_with_ML(model, LNS_log[-1], 
                                                                neighborhood_size=neighborhood_size, device=device, ML_info=ML_info,
                                                                wind_size=args.wind_size, feature_set=args.feature_set, greedy=greedy)

                logger(f"num of variables selected by ML: {len(destroy_variables)} in time {info_destroy_heuristic['ML_time']}", logfile = results_loc)

                # create sub MIP
                sub_mip = create_sub_mip(model, destroy_variables,  LNS_log[-1]['best_primal_sol'])

                # solve sub MIP
                logger("Solving sub MIP ...", logfile = results_loc)
                scip_solve_destroy_config = {'limits/time' : args.sub_time_limit}
                status, log_entry = scip_solve(sub_mip, scip_config=scip_solve_destroy_config, 
                                                incumbent_solution=best_sol, primal_bound=primal_bound,
                                                prev_LNS_log=LNS_log[-1])

                logger(f"step {s} repair variables in time {log_entry['iteration_time']}", logfile = results_loc)
                logger(f"step {s} repair variables in solving time {log_entry['solving_time']}", logfile = results_loc)

                improvement = abs(primal_bound - log_entry["primal_bound"])
                improved = (obj_sense * (primal_bound - log_entry["primal_bound"]) > 1e-5)
                LNS_log.append(log_entry)

                # change neigborhood size and greedy
                if improved == False:

                    logger(f"No improvement. Adjusted neighborhood_size: {neighborhood_size}, greedy: {greedy}", logfile = results_loc)

                    
                    if round(neighborhood_size * args.adaptive) < round(num_int_var * 0.5):
                        neighborhood_size = round(neighborhood_size * args.adaptive)
                    
                    else:
                        neighborhood_size = round(num_int_var * 0.5)
                        if greedy:
                            greedy = False

                    print(neighborhood_size)
                else:
                    relaxation_value = info_destroy_heuristic["LB_LP_relaxation_solution"]
                    incumbent_solution = []
                    LB_relaxation_solution = []

                    for var in int_var:
                        LB_relaxation_solution.append(relaxation_value.value(var))
                        incumbent_solution.append(log_entry["var_index_to_value"][var.name])
                    
                    LB_relaxation_history.append(LB_relaxation_solution)
                    incumbent_history.append(incumbent_solution)
                    improvement_history.append(improvement)

                iteration_end_time = time.monotonic()
                iteration_time = iteration_end_time - iteration_start_time
                runtime_used += iteration_time
                logger(f"Problem {instance_id}: Finished LNS step {s}: obj_val = {log_entry['primal_bound']} in iteration time {iteration_time}", logfile = results_loc)
                print(f"Problem {instance_id}: Finished LNS step {s}: obj_val = {log_entry['primal_bound']} in iteration time {iteration_time}")

                if runtime_used >= self.time_limit: break
                
                time_points.append(runtime_used)
                primal_bounds.append(log_entry["primal_bound"])
                variable_assignments.append(log_entry["var_index_to_value"])

            # End LNS
            end_run = time.monotonic()
            total_time = end_run - start_run

            # 记录初始解信息（第一个时间点的数据）
            if time_points and primal_bounds:
                logger(
                    f"Problem {instance_id}: initial solution obj = {primal_bounds[0]}, "
                    f"found in time {time_points[0]:.2f}s",
                    logfile=results_loc
                )

            # 记录最优解信息（从所有时间点中找最优）
            
            if primal_bounds:
                # best_idx = np.argmax(primal_bounds)  # 假设是最小化问题
                if self.problem == "cauctions" or self.problem == "indset":
                    best_idx = np.argmax(primal_bounds)  # 假设是最小化问题
                elif self.problem == "setcover" or self.problem == "mvc":
                    best_idx = np.argmin(primal_bounds)  # 假设是最大化问题
                else:
                    raise ValueError(f"Unknown problem type: {self.problem}")
                logger(
                    f"Problem {instance_id}: best primal bound {primal_bounds[best_idx]}, "
                    f"found at time {time_points[best_idx]:.2f}s, "
                    f"total time {time_points[-1]:.2f}s",
                    logfile=results_loc
                )
            # logger(f"Problem {instance_id}: initial solution obj = {log_entry['primal_bound']}, found in time {log_entry['run_time']}", logfile = results_loc)
            # logger(f"Problem {instance_id}: best primal bound {LNS_log[-1]['primal_bound']} in total time {total_time}", logfile = results_loc)
            # # out data

            logger(
                f"type: done, episode: {episode}, instance: {instance}, seed: {seed}, time: {total_time}, time_points: {time_points}, primal_bound: {primal_bounds}", 
                logfile = results_loc)
            
            model.freeProb()
            metric_log = {
                "instance_id": instance_id,
                "method": self.method,
                "time_points": time_points,
                "primal_bounds": primal_bounds,
                "variable_assignments": variable_assignments,
                "final_obj": LNS_log[-1]["primal_bound"],
                "total_runtime": total_time,
                "num_steps": len(primal_bounds),
            }

        elif self.method == "lns_GBDT":
            
            pickle_path = instance
            # pair_path = Path(str(instance.parent).replace("pickle_data", "pair_data"))/instance.name.replace("instance_", "pair").replace(".pickle", ".pkl")
            instance = Path(instance)  # 确保 instance 是 Path 对象
            # print("instance", instance)
            pair_path = Path(str(instance).replace("pickle_data", "pair_data").replace("instance_", "pair").replace(".pickle", ".pkl"))
            
            # pair_path = instance.parent.with_name("pair_data") / instance.name.replace("instance_", "pair").replace(".pickle", ".pkl")
            # print("pair_path", pair_path)

            LNS_log, times, obj_lis = optimize(pair_path,
             pickle_path,
             args.model_path,
             args.pre_model_path,
             0.5,
             self.time_limit,
             0.5,
             self.device,
             results_loc,
             instance_id,
             self.problem)

            metric_log = {
                "instance_id": instance_id,
                "method": self.method,
                "time_points": times,
                "primal_bounds": obj_lis,
                "variable_assignments": "",
                "final_obj": LNS_log[-1]["primal_bound"],
                "total_runtime": times[-1],
                "num_steps": len(obj_lis),
            }
        elif self.method == "lns_RL":
            gnnTester = GNNPolicyTester(device, self.data_path, self.method, self.problem, 'gnn',
                'gnn_critic', noise_type = None,
                normalize_returns = False,
                normalize_observations = False,
                load_path = None,
                model_dir = self.model_path,
                param_noise_adaption_interval = 30,
                exploration_strategy = 'relpscost')
            
            # print("model_path", self.model_path)

            LNS_log, times, obj_lis = gnnTester.test(instance, nb_epochs = 20,
                observation_range = (-np.inf, np.inf), 
                action_range = (0.2, 0.8),
                return_range = (-np.inf, np.inf),
                reward_scale = 1.0, critic_l2_reg = 1e-2,
                actor_lr = 1e-5,
                critic_lr = 1e-5,
                popart = False,
                gamma = 0.99,  
                clip_norm = None,
                nb_eval_steps = 1000, #20 #50  
                batch_size = 300, # per MPI worker  64 32  64   128   64  128 300
                tau = 0.01, 
                time_limit = 2,
                eval_val = 0,
                seed = 0,
                results_loc = results_loc,
                instance_id = instance_id)

            metric_log = {
                "instance_id": instance_id,
                "method": self.method,
                "time_points": times,
                "primal_bounds": obj_lis,
                "variable_assignments": "",
                "final_obj": LNS_log[-1]["primal_bound"],
                "total_runtime": times[-1],
                "num_steps": len(obj_lis),
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")
        # base_dir = Path(self.path_save_results)
        # base_dir.mkdir(parents=True, exist_ok=True)
        # csv_path = os.path.join(self.path_save_results, f"{self.task}_{self.problem}_{instance_id}_results.csv")
        # # CSV 文件路径
        # with open(csv_path, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["time_point", "primal_bound"])
        #     for t, val in zip(time_points, primal_bounds):
        #         writer.writerow([t, val])
        # print(f"Saved CSV for instance {instance_id} at {csv_path}")

        # # 画图
        # plt.figure()
        # plt.plot(time_points, primal_bounds, marker='o')
        # plt.xlabel("Time")
        # plt.ylabel("Primal Bound")
        # plt.title(f"Instance {instance_id} Optimization Progress")
        # plt.grid(True)

        # # 保存图像文件
        # img_path = base_dir / f"{self.task}_{self.problem}_{instance_id}_result.png"
        # plt.savefig(img_path)
        # plt.close()
        # print(f"Saved plot for instance {instance_id} at {img_path}")

        return LNS_log, metric_log
    
    def send_orders(self, episode, instance, device, results_loc, args):

        rng = np.random.RandomState(self.seed)

        orders_queue = []

        seed = rng.randint(2**32)
        orders_queue = [episode, instance, device, seed, results_loc, args]
        
        return orders_queue

    def set_policy(self, task, policy, device,
                    model_path, pre_model_path):
        
        self.task = task
        self.method = policy
        self.device = device
        self.model_path = model_path
        self.pre_model_path = pre_model_path
        parent_dir = os.path.dirname(self.data_path)
        self.instance_path = os.path.join(self.data_path, 'instances', self.method, self.problem)
        self.path_save_results = os.path.join(parent_dir, 'results', task, self.method, self.problem)
        os.makedirs(self.path_save_results, exist_ok=True)

    # def test(self, size, n_instance, multi_instance, time_limit):
        
    #     self.time_limit = time_limit

    #     args = CLArgs()
        
    #     args.model_path = self.model_path
    #     args.pre_model_path = self.pre_model_path

    #     NDPolicy = NDPolicyTestEnv(0.1)

    #     instances_dir = os.path.join(self.instance_path, size) # instance path
    #     instances_files = os.listdir(instances_dir)
    #     # dir to keep samples temporarily; helps keep a prefect count
    #     episode = 0
    #     objective_list = []
        
    #     logfile = os.path.join(self.path_save_results, "log.txt")
        
    #     with open(logfile, 'a') as f:
    #         f.write("Log file created or opened successfully.\n")

    #     for instance in instances_files:
    #         instance = os.path.join(instances_dir, instance)
    #         orders_queue = self.send_orders(episode, instance, self.device, logfile, args)
    #         episode += 1
    #         results_LNS = self.test_single_instance(orders_queue, NDPolicy)
    #         objective = results_LNS[-1]['primal_bound']
    #         objective_list.append(objective)
        
    #     print("mean results:", self.task, self.problem, size, np.mean(objective_list))

    def numeric_key(self, fname):
        nums = re.findall(r'\d+', fname)
        return int(nums[-1]) if nums else fname

    def test(self, size, n_instance, multi_instance, time_limit):

        self.time_limit = time_limit

        # 构造 args 与 NDPolicy
        args = CLArgs()
        args.model_path = self.model_path
        args.pre_model_path = self.pre_model_path

        NDPolicy = NDPolicyTestEnv(0.1)

        # 实例列表
        
        if self.method == "lns_GBDT":
            instances_dir = os.path.join(self.data_path, 'instances', self.method, 'pickle_data', self.problem, size)
        else:
            instances_dir = os.path.join(self.instance_path, size)

        
        instances_files = sorted(os.listdir(instances_dir),key=self.numeric_key)[:n_instance]  # 如果只跑前 n_instance 个

        # 确保主日志存在
        logfile = os.path.join(self.path_save_results, "log_test.txt")
        with open(logfile, 'a') as f:
            f.write("Log file created or opened successfully.\n")

        # 构造 (episode, filename) 列表
        episode_instances = list(enumerate(instances_files))

        print("Starting to run instances...")
        print("Total number of instances:", len(episode_instances))
        print("instances_files:", instances_files)

        worker = partial(
            _worker_process,
            self,
            instances_dir=instances_dir,
            logfile=logfile,
            args=args,
            NDPolicy=NDPolicy
        )

        with multiprocessing.Pool(processes=multi_instance) as pool:
            results = pool.map(worker, episode_instances)

        return results