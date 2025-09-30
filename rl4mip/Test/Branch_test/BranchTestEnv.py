import os
import sys
import numpy as np
import time
import datetime
import pickle
import pyscipopt as scip
import torch
from torch_scatter import scatter_mean, scatter_sum
from torch_scatter import scatter_max as scatter_max_raw, scatter_min as scatter_min_raw
import pandas as pd
import importlib

from rl4mip.Trainer.Branch_model.GNN import GNNPolicy
from .test_utils import init_scip_params, log, get_test_instances, get_results_statistic, _preprocess
from torch.multiprocessing import Process, set_start_method, Queue, Value
from functools import partial
from queue import Empty

def scatter_min(src, index):
    return scatter_min_raw(src, index)[0]

def scatter_max(src, index):
    return scatter_max_raw(src, index)[0]


# 嵌套函数不能用多线程，因此需要修改代码
# def get_symb_policy(model_path):
#     with open(model_path, "r") as txt:
#         expression = next(txt)
#     def get_scores(inputs):
#         result = torch.nn.functional.tanh(eval(expression))
#         return result
#     return get_scores
class SymbPolicy:
    def __init__(self, model_path):
        with open(model_path, "r") as txt:
            self.expression = next(txt)
    def get_scores(self, inputs):
        result = torch.nn.functional.tanh(eval(self.expression))
        return result
def get_symb_policy(model_path):
    policy = SymbPolicy(model_path)
    return policy.get_scores  # 返回绑定方法


# 嵌套函数不能用多线程，因此需要修改代码
# def get_graph_policy(model_path):
#     with open(model_path, "r") as txt:
#         expression = next(txt)
#     variable_allocation_exp, calculation_exp = expression.split(";;")
#     def get_logits(constraint, cv_edge_index, edge_attr, variable):
#         c_edge_index, v_edge_index = cv_edge_index
#         exec(variable_allocation_exp)
#         result = eval(calculation_exp)
#         return result
#     return get_logits
class GraphPolicy:
    def __init__(self, model_path):
        with open(model_path, "r") as txt:
            expression = next(txt)
        self.variable_allocation_exp, self.calculation_exp = expression.split(";;")
    def get_logits(self, constraint, cv_edge_index, edge_attr, variable):
        c_edge_index, v_edge_index = cv_edge_index
        exec(self.variable_allocation_exp)
        result = eval(self.calculation_exp)
        return result
def get_graph_policy(model_path):
    policy = GraphPolicy(model_path)
    return policy.get_logits  # 返回绑定方法

class PolicyBranching(scip.Branchrule):

    def __init__(self, policy, device, 
                 model_path=None, teacher_model_path=None,
                 model_name='film-pre', teacher_model='gnn_policy'):
        
        super().__init__()
        self.policy = policy
        self.device = device

        log(f"testing policy model {model_path} ...")

        #### load policy ####
        if self.policy == 'gnn':
            model = GNNPolicy().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policymodel = model
        
        elif self.policy == 'hybrid':
            log(f"model name is {model_name}")
            sys.path.insert(0, os.path.abspath(f'rl4mip/Trainer/Branch_model/hybrid_model/{model_name}'))
            from rl4mip.Trainer.Branch_model.hybrid_model.film import model
            # import model
            importlib.reload(model)
            policy_model = model.Policy().to(self.device)
            policy_model.eval()
            del sys.path[0]
            policy_model.restore_state(model_path)
            self.policy_get_root_params = policy_model.get_params
            self.policy_predict = policy_model.predict

            self.teacher = None
            if "-pre" in model_name:
                sys.path.insert(0, os.path.abspath(f'rl4mip/Trainer/Branch_model/hybrid_model/{teacher_model}'))
                import model
                importlib.reload(model)
                self.teacher = model.GCNPolicy().to(self.device)
                self.teacher.eval()
                del sys.path[0]
                self.teacher.restore_state(teacher_model_path)

        elif self.policy == 'symb':
            self.policymodel = get_symb_policy(model_path)
            
        elif self.policy == 'graph':
            self.policymodel = get_graph_policy(model_path)

        
    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}

    def branchexeclp(self, allowaddcons):

        # SCIP internal branching rule
        if self.policy == 'relpscost':
            result = self.model.executeBranchRule(self.policy, allowaddcons)

        # custom policy branching
        else:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            if len(candidate_vars) == 1:
                best_var = candidate_vars[0]

            elif self.policy in ['gnn', 'graph']:

                var_features, edge_features, cons_features, _ = self.model.getBipartiteGraphRepresentation()
            
                indices = [[row[1] for row in edge_features],[row[0] for row in edge_features]]
                values = [row[2] for row in edge_features]
                mean = sum(values) / len(values)
                squared_diffs = [(x - mean) ** 2 for x in values]
                variance = sum(squared_diffs) / len(squared_diffs)
                std = variance ** 0.5 + 1e-4
                normalized_values = [(x - mean) / std for x in values]
                edge_features_dic = {'indices':indices, 'values':normalized_values}

                cons_features = torch.as_tensor(cons_features, dtype=torch.float32, device=self.device)
                edge_indices = torch.as_tensor(edge_features_dic['indices'], dtype=torch.long, device=self.device)
                edge_values = torch.as_tensor(edge_features_dic['values'], dtype=torch.float32, device=self.device)
                if self.policy == 'gnn':
                    edge_values = edge_values.reshape(-1, 1)
                var_features = torch.as_tensor(var_features, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    observation = ( cons_features,
                                    edge_indices,
                                    edge_values,
                                    var_features )

                    logits = self.policymodel(*observation)
                    logits_np = logits.cpu().numpy()

                    candidate_scores = logits_np[candidate_mask]
                    best_var = candidate_vars[candidate_scores.argmax()]

            elif self.policy == 'hybrid':
                var_features, edge_features, cons_features, _ = self.model.getBipartiteGraphRepresentation()
                indices = [[row[1] for row in edge_features],[row[0] for row in edge_features]]
                values = [row[2] for row in edge_features]
                mean = sum(values) / len(values)
                squared_diffs = [(x - mean) ** 2 for x in values]
                variance = sum(squared_diffs) / len(squared_diffs)
                std = variance ** 0.5 + 1e-4
                normalized_values = [(x - mean) / std for x in values]
                edge_features_dic = {'indices':indices, 'values':normalized_values}
                var_features = np.array(var_features)
                cons_features = np.array(cons_features)

                if self.model.getNNodes() == 1:
                    n_cons = torch.as_tensor(cons_features.shape[0], dtype=torch.int32, device=self.device)
                    n_vars = torch.as_tensor(var_features.shape[0], dtype=torch.int32, device=self.device)
                    cons_features = torch.as_tensor(cons_features, dtype=torch.float32, device=self.device)
                    edge_indices = torch.as_tensor(edge_features_dic['indices'], dtype=torch.long, device=self.device)
                    edge_values = torch.as_tensor(edge_features_dic['values'], dtype=torch.float32, device=self.device)
                    edge_values = edge_values.reshape(-1, 1)
                    var_features = torch.as_tensor(var_features, dtype=torch.float32, device=self.device)

                    state = (   cons_features,
                                edge_indices,
                                edge_values,
                                var_features,
                                n_cons,
                                n_vars  )
                    with torch.no_grad():
                        if self.teacher is not None:
                            root_feats, _ = self.teacher(state)
                            state = root_feats
                        self.root_params = self.policy_get_root_params(state)

                candi_features, _, _ = self.model.getBranchFeaturesRepresentation(candidate_vars)
                var_feats = np.array(candi_features)
                var_feats = _preprocess(var_feats, mode='min-max-2')
                var_feats = torch.as_tensor(var_feats, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    var_logits = self.policy_predict(var_feats, self.root_params[candidate_mask]).cpu().numpy()

                best_var = candidate_vars[var_logits.argmax()]
            
            elif self.policy == 'symb':
                candi_features, _, _ = self.model.getBranchFeaturesRepresentation(candidate_vars)
                var_feats = np.array(candi_features)
                var_feats = _preprocess(var_feats, mode='min-max-1')
                var_feats = torch.as_tensor(var_feats, dtype=torch.float32).to(self.device)
                var_scores = self.policymodel(var_feats)
                var_scores = var_scores.cpu().numpy()
                best_var = candidate_vars[var_scores.argmax()]

            else:
                raise NotImplementedError

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}


def init_scip_params_4_node(model, seed=9, time_limit=3600):
    # print("使用node的参数")
    model.setIntParam('randomization/permutationseed',seed) 
    model.setIntParam('randomization/randomseedshift',seed)
    model.setParam('constraints/linear/upgrade/logicor',0)
    model.setParam('constraints/linear/upgrade/indicator',0)
    model.setParam('constraints/linear/upgrade/knapsack', 0)
    model.setParam('constraints/linear/upgrade/setppc', 0)
    model.setParam('constraints/linear/upgrade/xor', 0)
    model.setParam('constraints/linear/upgrade/varbound', 0)
    model.setRealParam('limits/time', time_limit)
    model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()

def init_scip_params_4_branch(model, time_limit, gap_limit, seed, heuristics=True, presolving=True, separating=True, conflict=True):
    # print("使用branch的参数")
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
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

    model.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
    model.setRealParam('limits/time', time_limit)
    model.setRealParam('limits/gap', gap_limit)


def init_scip_params_4_cut(model, time_limit):
    # print("使用cut的参数")
    ""

def init_scip_params_4_intersection(model, seed, time_limit):
    # print("使用node、branch、cut参数的交集")
    model.setIntParam('randomization/permutationseed',seed) 
    model.setIntParam('randomization/randomseedshift',seed)

    model.setRealParam('limits/time', time_limit)

def init_scip_params_new(model, seed, time_limit, heuristics=True, presolving=True, separating=True, conflict=True):
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
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

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


def get_record_file(now_time, instances_type, instance, size, method):
    # save_dir = os.path.join('/data/home/zdhs0047/zdhs0047_src_data/benchmark/benchmark_draft-main/', f'test_branch/{instances_type}/{size}/{method}/{now_time}/')
    save_dir = os.path.join(os.path.abspath(''), f'test_branch/efficiency_data/{instances_type}/{size}/{method}/{now_time}/')
    
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        ""
    instance = str(instance).split('/')[-1]
    file = os.path.join(save_dir, instance.replace('.lp', '.csv'))
    return file

def get_optimal_data_file(now_time, instances_type, instance, size, method):
    save_dir = os.path.join(os.path.abspath(''), f'test_branch/optimal_data/{instances_type}/{size}/{method}/{now_time}/')
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        ""
    instance = str(instance).split('/')[-1]
    file = os.path.join(save_dir, instance.replace('.lp', '.txt'))
    return file

def get_sol_data_file(now_time, instances_type, instance, size, method):
    save_dir = os.path.join(os.path.abspath(''), f'test_branch/solution_data/{instances_type}/{size}/{method}/{now_time}/')
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        ""
    instance = str(instance).split('/')[-1]
    file = os.path.join(save_dir, instance.replace('.lp', '.sol'))
    return file

def record_stats_instance(now_time, stime, nnodes, pdintegral, gap, optimal_solution, optimal_value, instances_type, instance, size, method, worker_id):
    save_file = get_record_file(now_time, instances_type, instance, size, method)

    last_three_dirs = os.sep.join(instance.split(os.sep)[-3:])
    # print("Process: ", worker_id, " ", last_three_dirs, ": ", np.array([stime, nnodes, pdintegral, gap]))
    np.savetxt(save_file, np.array([stime, nnodes, pdintegral, gap]), delimiter=',')

    file_optimal_data = get_optimal_data_file(now_time, instances_type, instance, size, method) 
    from pathlib import Path
    p = Path(file_optimal_data)
    p.parent.mkdir(parents=True, exist_ok=True)   # 确保目录存在
    with p.open("w", encoding="utf-8") as f:
        f.write("optimal_solution:\n" + "".join(map(str, optimal_solution)) + "\n")
        f.write(f"optimal_value:\n{optimal_value}\n")





def get_mean(now_time, instances_type, instances, size, method, stat_type):
    res = []
    n = 0
    means = dict()
    stat_idx = [ 'time', 'nnode', 'pd_integral', 'gap'].index(stat_type)
    for instance in instances:
        try:
            file = get_record_file(now_time, instances_type, instance, size, method)
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


def display_stats(now_time, instances_type, size, instances_list, method):
    time_mean, num_res, _, time_dev =  get_mean(now_time, instances_type, instances_list, size, method, 'time')
    nnode_mean, _, _, nnode_dev = get_mean(now_time, instances_type, instances_list, size, method, 'nnode')
    pdi_mean = get_mean(now_time, instances_type, instances_list, size, method, 'pd_integral')[0]
    gap_mean = get_mean(now_time, instances_type, instances_list, size, method, 'gap')[0]

    return time_mean, time_dev, nnode_mean, nnode_dev, gap_mean, pdi_mean, num_res

def test_multiprocess(size, method, now_time, instances, seed, brancher, policy, problem, time_limit=3600, gap_limit=0.0, scip_para=2, worker_id=0):
    for instance_path in instances:
        result = test_single(now_time, size, method, instance_path, seed, brancher, policy, problem, worker_id, time_limit, gap_limit, scip_para)
        if result["status"] != 'optimal':
            log(f"instance {instance_path} does not obtain optimal, status is: {result['status']}")

def test_single(now_time, size, method, instance_path, seed, brancher, policy, problem, worker_id, time_limit=300, gap_limit=0.0, scip_para=2):
    m = scip.Model()
    m.hideOutput()
    m.readProblem(instance_path)
    torch.manual_seed(seed)

    if scip_para == 1:
        # 使用和 node 一样的参数
        seed = 9
        init_scip_params_4_node(m, seed, time_limit)
    elif scip_para == 2:
        # 使用和 branch 一样的参数
        init_scip_params_4_branch(m, time_limit, gap_limit, seed)
    elif scip_para == 3:
        # 使用和 cut 一样的参数
        init_scip_params_4_cut(m, time_limit)
    elif scip_para == 4:
        # 使用的参数的交集
        init_scip_params_4_intersection(m, seed, time_limit)
    elif scip_para == 5:
        seed = 0
        init_scip_params_new(m, seed, time_limit)
    else:
        # scip_para 不是有效值，抛出异常
        raise ValueError(f"Invalid value for scip_para: {scip_para}. Expected 1, 2, 3, or 4.")


    m.includeBranchrule(
        branchrule=brancher,
        name='',
        desc="GNN branching policy.",
        priority=666666, maxdepth=-1, maxbounddist=1)
    
    walltime = time.perf_counter()
    proctime = time.process_time()

    m.optimize()

    walltime = time.perf_counter() - walltime
    proctime = time.process_time() - proctime

    stime = m.getSolvingTime()
    nnodes = m.getNNodes()
    nlps = m.getNLPs()
    gap = m.getGap()
    pdintegral = m.getPrimalDualIntegral()
    status = m.getStatus()
    ndomchgs = brancher.ndomchgs
    ncutoffs = brancher.ncutoffs

    # 获取最优解
    variables = m.getVars()
    optimal_solution = [m.getSolVal(m.getBestSol(), var) for var in variables]
    # 获取最优值
    optimal_value = float(m.getObjVal())

    sol_file = get_sol_data_file(now_time, problem, instance_path, size, method)
    m.writeBestSol(sol_file)

    # print("AAAAA:最优值，最优解")
    # print(optimal_solution)
    # print(optimal_value)

    result = {  'policy': policy,
                'seed': seed,
                'instance': instance_path,
                'nnodes': nnodes,
                'nlps': nlps,
                'stime': stime,
                'gap': gap,
                'PDintegral':pdintegral,
                'status': status,
                'ndomchgs': ndomchgs,
                'ncutoffs': ncutoffs,
                'walltime': walltime,
                'proctime': proctime
                }
    m.freeProb()
    # log(result)
    record_stats_instance(now_time, stime, nnodes, pdintegral, gap, optimal_solution, optimal_value, problem, instance_path, size, method, worker_id) # instances_type, instance, size
    return result


class TaskManager:
    def __init__(self, now_time, size, method, instances, seed, brancher, policy, problem, time_limit, gap_limit, scip_para, n_cpu):

        self.now_time = now_time
        self.size = size
        self.method = method
        self.instances = instances
        self.time_limit = time_limit
        self.gap_limit = gap_limit
        self.scip_para = scip_para
        self.seed = seed
        self.brancher = brancher
        self.policy = policy
        self.problem = problem
        self.n_cpu = n_cpu


        
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.error_queue = Queue()
        self.completed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        
    def worker(self, worker_id):
        """工作进程函数"""
        while True:
            try:
                # 从任务队列获取任务，超时1秒
                task_data = self.task_queue.get(timeout=1)
                if task_data is None:  # 毒丸，表示没有更多任务
                    break
                instance_idx, instance = task_data
                # 执行原来的函数
                test_multiprocess(
                    size=self.size,
                    method=self.method,
                    now_time=self.now_time,
                    instances=[instance],
                    seed=self.seed,
                    brancher=self.brancher,
                    policy=self.policy,
                    problem=self.problem,
                    time_limit=self.time_limit,
                    gap_limit=self.gap_limit,
                    scip_para=self.scip_para,
                    worker_id=worker_id,
                )
            except Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                break

    def run(self):
        """执行所有任务"""
        # 将所有任务放入队列
        for i, instance in enumerate(self.instances):
            self.task_queue.put((i, instance))
        
        # 启动工作进程
        processes = []
        for i in range(self.n_cpu):
            p = Process(target=self.worker, args=(i,))
            p.start()
            processes.append(p)
        
        # 发送停止信号给所有工作进程
        for _ in range(self.n_cpu):
            self.task_queue.put(None)
        
        # 等待所有进程结束
        for p in processes:
            p.join()



class BranchPolicyTestEnv:
    """Environment for test MIP branching policies"""
    
    def __init__(self, problem, data_path, seed=0,):
        self.problem = problem
        self.data_path = data_path
        self.seed = seed
    
    def set_policy(self, policy, device, 
                    model_path=None, teacher_model_path=None,
                    model_name='film-pre', teacher_model='gnn_policy',scip_para=2, n_cpu=16):
        self.policy = policy
        self.device = device
        self.model_path = model_path
        self.model_name = model_name
        self.teacher_model = teacher_model
        self.scip_para=scip_para
        self.n_cpu = n_cpu
        self.brancher = PolicyBranching(policy, device, model_path, teacher_model_path, model_name, teacher_model)


    
    # def test_single(self, now_time, size, method, instance_path, time_limit=300, gap_limit=0.0, scip_para=2):
    #     m = scip.Model()
    #     m.hideOutput()
    #     m.readProblem(instance_path)
    #     torch.manual_seed(self.seed)

    #     if scip_para == 1:
    #         # 使用和 node 一样的参数
    #         seed = 9
    #         init_scip_params_4_node(m, seed)
    #     elif scip_para == 2:
    #         # 使用和 branch 一样的参数
    #         init_scip_params_4_branch(m, time_limit, gap_limit, self.seed)
    #     elif scip_para == 3:
    #         # 使用和 cut 一样的参数
    #         init_scip_params_4_cut(m)
    #     elif scip_para == 4:
    #         # 使用的参数的交集
    #         init_scip_params_4_intersection(m, self.seed)
    #     else:
    #         # scip_para 不是有效值，抛出异常
    #         raise ValueError(f"Invalid value for scip_para: {scip_para}. Expected 1, 2, 3, or 4.")


    #     m.includeBranchrule(
    #         branchrule=self.brancher,
    #         name='',
    #         desc="GNN branching policy.",
    #         priority=666666, maxdepth=-1, maxbounddist=1)
        
    #     walltime = time.perf_counter()
    #     proctime = time.process_time()

    #     m.optimize()

    #     walltime = time.perf_counter() - walltime
    #     proctime = time.process_time() - proctime

    #     stime = m.getSolvingTime()
    #     nnodes = m.getNNodes()
    #     nlps = m.getNLPs()
    #     gap = m.getGap()
    #     pdintegral = m.getPrimalDualIntegral()
    #     status = m.getStatus()
    #     ndomchgs = self.brancher.ndomchgs
    #     ncutoffs = self.brancher.ncutoffs

    #     result = {  'policy': self.policy,
    #                 'seed': self.seed,
    #                 'instance': instance_path,
    #                 'nnodes': nnodes,
    #                 'nlps': nlps,
    #                 'stime': stime,
    #                 'gap': gap,
    #                 'PDintegral':pdintegral,
    #                 'status': status,
    #                 'ndomchgs': ndomchgs,
    #                 'ncutoffs': ncutoffs,
    #                 'walltime': walltime,
    #                 'proctime': proctime
    #                 }
    #     m.freeProb()
    #     # log(result)
    #     record_stats_instance(now_time, stime, nnodes, pdintegral, gap, self.problem, instance_path, size, method) # instances_type, instance, size
    #     return result
    
    # def test_multiprocess(self, size, method, now_time, instances, time_limit=300, gap_limit=0.0):
    #     for instance_path in instances:
    #         result = self.test_single(now_time, size, method, instance_path, time_limit, gap_limit, self.scip_para)
    #         if result["status"] != 'optimal':
    #             log(f"instance {instance_path} does not obtain optimal, status is: {result['status']}")


    
    def test(self, size='small', n_instance=2, time_limit=300, gap_limit=0.0):
        # set_start_method('spawn', force=True)
        # set_start_method('spawn')
        problem_size = f'{self.problem}_{size}'
        log(f"testing problem size {size} ...")
        instance_list = get_test_instances(self.data_path, problem_size, n_instance)
        result_list = []

        # for instance_path in instance_list:
        #     result = self.test_single(instance_path, time_limit, gap_limit, self.scip_para)
        #     if result["status"] != 'optimal':
        #         log(f"instance {instance_path} does not obtain optimal, status is: {result['status']}")
            
        #     result_list.append(result)


        if 'gnn' in self.model_path:
            method = 'gnn'
        elif 'symb' in self.model_path:
            method = 'symb'
        elif 'graph' in self.model_path:
            method = 'graph'
        elif 'hybrid' in self.model_path:
            method = 'hybrid'
        else:
            #抛出异常
            raise ValueError(f"Unrecognized model path: {self.model_path}. The model path must contain one of 'gnn', 'symb', or 'graph'.")

        # print("进程：", distribute(n_instance, self.n_cpu))
        # 多线程
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # processes = [Process(name=f"worker {p}", 
        #                 target=partial(self.test_multiprocess,
        #                                 now_time=now_time,
        #                                 size=size,
        #                                 method=method,
        #                                 instances=instance_list[p1:p2],
        #                                 time_limit=time_limit,
        #                                 gap_limit=gap_limit,
        #                                 ))
        #             for p,(p1,p2) in enumerate(distribute(n_instance, self.n_cpu))] # self.n_cpu

        # # 'set_start_method('spawn')' 需要放在 if __name__ == "__main__": 下面
        # a = list(map(lambda p: p.start(), processes)) #run processes
        # b = list(map(lambda p: p.join(), processes)) #join processes
        # torch.cuda.empty_cache()


        # test_multiprocess(
        #     size=size,
        #     method=method,
        #     now_time=now_time,
        #     instances=instance_list,
        #     seed=self.seed,
        #     brancher=self.brancher,
        #     policy=self.policy,
        #     problem=self.problem,
        #     time_limit=time_limit,
        #     gap_limit=gap_limit,
        #     scip_para=self.scip_para,
        #     worker_id=0,
        # )

        # exit(0)

        # 创建TaskManager并运行
        task_manager = TaskManager(
                                    now_time=now_time,
                                    size=size,
                                    method=method,
                                    instances=instance_list,
                                    seed=self.seed,
                                    brancher=self.brancher,
                                    policy=self.policy,
                                    problem=self.problem,
                                    time_limit=time_limit,
                                    gap_limit=gap_limit,
                                    scip_para=self.scip_para,
                                    n_cpu=self.n_cpu
        )
        task_manager.run()



        # 计算结果
        time_mean, time_dev, nnode_mean, nnode_dev, gap_mean, pdi_mean, num_res = display_stats(now_time, self.problem, size, instance_list, method)
        
        # 检查目录是否存在
        data_path = os.path.join(self.data_path, "instances")
        directory = os.path.join(data_path, self.problem, size+"_test")
        print("directory = ", directory)
        # 将路径按 '/' 分割成一个列表
        path_parts = directory.strip('/').split('/')
        # 获取最后两个部分
        last_two_parts = '/'.join(path_parts[-2:])
        # print("last_two_parts = ", last_two_parts)

        # result_dict, df = get_results_statistic(result_list)

        test_branch_efficiency_data = os.path.join(os.path.abspath(''),  f'test_branch/efficiency_data/{self.problem}/{size}/{method}/{now_time}/')
        test_branch_optimal_data = os.path.join(os.path.abspath(''),  f'test_branch/optimal_data/{self.problem}/{size}/{method}/{now_time}/') 
        test_branch_sol_data = os.path.join(os.path.abspath(''),  f'test_branch/sol_data/{self.problem}/{size}/{method}/{now_time}/')
        
        data_to_append = [
            "==============================================================\n",
            f"now_time: {now_time}\n",
            f"instances: {last_two_parts}\n",
            f"线程数量: {self.n_cpu}\n",
            f"方法: {method}\n",
            f"参数(1:node,2:branch,3:cut,4:交集,5:默认): {self.scip_para}\n",
            f"待求解数量: {len(instance_list)}\n",
            f"成功求解数量: {num_res}\n",
            f"求解时间: {time_mean:.2f}±{time_dev:.2f}秒\n",
            f"节点数: {nnode_mean:.2f}±{nnode_dev:.2f}\n",
            f"平均GAP: {gap_mean:.2f}\n",
            f"平均PDI: {pdi_mean:.2f}\n",
            f"求解效率记录保存在: {test_branch_efficiency_data}\n",
            # f"最优值与最优解保存在: {test_branch_optimal_data}\n",
            f"sol文件保存在: {test_branch_sol_data}\n",
            "==============================================================\n"
        ]
        with open('branch_output.txt', 'a', encoding='utf-8') as file:
            file.writelines(data_to_append)
        
        return ''.join(data_to_append)





