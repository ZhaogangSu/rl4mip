import os
import numpy as np
import shutil
import stat
import time
import pyscipopt as scip
import torch
import sys
from ml4co.DataCollector.lns_data.IL_data.utils import logger, init_scip_params, scip_solve, create_neighborhood_with_LB, create_sub_mip, get_perturbed_samples, make_obs
from ml4co.DataCollector.lns_data.IL_data.bipartite_graph import BipartiteGraph
from ml4co.DataCollector.lns_data.IL_data.bipartite_graph_dataset import BipartiteGraphDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def make_samples(in_queue):

    out_queue = []
    
    for i in range(len(in_queue)):

        episode, task, type, instance, seed, args, outdir = in_queue[i]

        instance_id = instance.split('/')[-1].split(".lp")[0]
        print(instance_id)

        os.makedirs(f'./collect_res/{task}/{args.problem}/{type}', exist_ok=True)

        results_loc = f'./collect_res/{task}/{args.problem}/{type}/{instance_id}.txt'

        model = scip.Model()
        model.setIntParam('display/verblevel', 0)
        model.setIntParam('timing/clocktype', 2)  # 1: CPU user seconds, 2: wall clock time
        model.readProblem(f'{instance}')

        # out_queue.put({
        #     "type":'start',
        #     "episode":episode,
        #     "instance":instance,
        #     "seed": seed
        # })

        # get features
        observation0 = make_obs(instance,seed)

        # use model to collect local branching actions
        init_scip_params(model, seed=seed, presolving=False)
        int_var = [v for v in model.getVars() if v.vtype() in ["BINARY", "INTEGER"]]
        objective_sense = model.getObjectiveSense()
        obj_sense = 1 if objective_sense == "minimize" else -1

        logger(f"Num of integer variables: {len(int_var)}", results_loc)
        if len(observation0["var_features"]) != len(int_var):
            logger('variable features error', results_loc)
            out_queue.put({
                "type": "failed",
                "episode": episode,
                "instance": instance,
                "seed": seed
                })
            continue

        # initial bgd
        filename = os.path.join(outdir, f'{instance_id}.db')
        database = BipartiteGraphDataset(filename)

        # find initial solution with SCIP in 10s
        scip_solve_init_config = {'limits/solutions' :10000, 'limits/time' : args.init_time_limit}
        status, log_entry = scip_solve(model, scip_config = scip_solve_init_config)
        if log_entry is None:
            logger(f'{instance} did not find initial solution',results_loc)
            out_queue.put({
                "type": "failed",
                "episode": episode,
                "instance": instance,
                "seed": seed
                })
            continue
        
        logger(f"initial solution obj = {log_entry['primal_bound']}, found in time {log_entry['run_time']}", results_loc)
        LNS_log = [log_entry]
        count_no_improve = 0

        # initialize incumbent_history with the initial solution
        incumbent_solution = []
        incumbent_history = []
        improvement_history = []
        LB_relaxation_history = []
        for var in int_var:        
            incumbent_solution.append(log_entry["var_index_to_value"][var.name])
        incumbent_history.append(incumbent_solution)

        logger("Start LNS iteration ...", results_loc)
        logger(f"Solving steps limit: {args.num_solve_steps}",results_loc)
        logger(f"Neighborhood size: {args.neighborhood_size}",results_loc)
        for s in range(args.num_solve_steps):
            iteration_start_time = time.monotonic()
            best_sol = LNS_log[-1]['best_primal_scip_sol']
            primal_bound = LNS_log[-1]['primal_bound']

            # use local branching to get destroy variables
            destroy_variables, info_destroy_heuristic = create_neighborhood_with_LB(model, LNS_log[-1], 
                neighborhood_size=args.neighborhood_size, time_limit=args.collect_time_limit, get_num_solutions=20)
            logger(f"num of variables selected by LB: {len(destroy_variables)} with obj {info_destroy_heuristic["LB_primal_solution"]}", results_loc)

            # create sub MIP
            sub_mip = create_sub_mip(model, destroy_variables,  LNS_log[-1]['best_primal_sol'])

            # solve sub MIP
            logger("Solving sub MIP ...", results_loc)
            scip_solve_destroy_config = {'limits/time' : args.sub_time_limit}
            status, log_entry = scip_solve(sub_mip, scip_config=scip_solve_destroy_config, 
                                            incumbent_solution=best_sol, primal_bound=primal_bound,
                                            prev_LNS_log=LNS_log[-1])
            logger(f"step {s} repair variables in time {log_entry['iteration_time']}",results_loc)

            improvement = abs(primal_bound - log_entry["primal_bound"])
            improved = (obj_sense * (primal_bound - log_entry["primal_bound"]) > 1e-5)
            LNS_log.append(log_entry)

            if improved == True:

                relaxation_value = info_destroy_heuristic["LB_LP_relaxation_solution"]
                assert relaxation_value is not None
                candidate_scores = []
                incumbent_solution = []
                LB_relaxation_solution = []

                for var in int_var:
                    if var.name in destroy_variables: 
                        candidate_scores.append(1) # 被改变的是1，不被改变的是0
                    else:
                        candidate_scores.append(0)

                    LB_relaxation_solution.append(relaxation_value.value(var))
                    incumbent_solution.append(log_entry["var_index_to_value"][var.name])
                
                LB_relaxation_history.append(LB_relaxation_solution)
                incumbent_history.append(incumbent_solution)
                improvement_history.append(improvement)
            
                # logger("Start collecting negative samples ...",results_loc)
                # negative_samples, negative_info, negative_labels = get_perturbed_samples(model, destroy_variables, LNS_log[-1], args.sub_time_limit, 90, int_var)
                # logger(f"Collected {len(negative_samples)} negative samples ...", results_loc)

                logger("Start collecting positive samples ...", results_loc)
                positive_samples = []
                positive_labels = []

                for i in range(len(info_destroy_heuristic["multiple_primal_bounds"])):
                    positive_sample = [0] * len(int_var)
                
                    for j, var in enumerate(int_var):
                        positive_sample[j] = info_destroy_heuristic["multiple_solutions"][var.name][i]
                    positive_samples.append(positive_sample)
                    obj_info = info_destroy_heuristic["multiple_primal_bounds"][i]
                    positive_labels.append( abs(obj_info[0] - obj_info[1]))
            
                logger(f"Collected {len(positive_samples)} positive samples ...", results_loc)

                # storage samples
                candidates = [str(var.name) for var in int_var]
                
                candidate_choice = None
                # for i in range(len(info_destroy_heuristic["multiple_primal_bounds"])):
                #     choice_mask = [0] * len(int_var)
                #     for j, var in enumerate(int_var):
                #         if info_destroy_heuristic["multiple_solutions"][var.name][i] == 1:
                #             choice_mask[j] = 1
                #     candidate_choice.append(choice_mask)

                # candidate_choice = torch.FloatTensor(candidate_choice)  # shape: [B, num_vars]

                # for sample in positive_samples:
                #     try:
                #         idx = sample.index(1)
                #     except ValueError:
                #         idx = -1  # 如果没有1，设置为-1，后续训练时要注意过滤或处理
                #     candidate_choice.append(idx)
                
                # candidate_choice = torch.LongTensor(candidate_choice)

                info = dict()
                info["num_positive_samples"] = len(positive_samples)
                info["positive_samples"] = positive_samples
                info["positive_labels"] = positive_labels

                # info["num_negative_samples"] = len(negative_samples)
                # info["negative_samples"] = negative_samples
                # info["negative_labels"] = negative_labels

                info["incumbent_history"] = incumbent_history
                info["improvement_history"] = improvement_history
                info["LB_relaxation_history"] = LB_relaxation_history
                info["neighborhood_size"] = args.neighborhood_size
                info["primal_bound"] = log_entry["primal_bound"]
                info["LB_runtime"] = log_entry["run_time"]

                candidate_scores = torch.LongTensor(np.array(candidate_scores, dtype=np.int32))
                constraint_features = torch.FloatTensor(np.array(observation0["cons_features"], dtype=np.float64))
                variable_features = torch.FloatTensor(np.array(observation0["var_features"], dtype=np.float64))
                edge_indices = torch.LongTensor(np.array(observation0["edge_features"]["indices"], dtype=np.int32))
                edge_features = torch.FloatTensor(np.expand_dims(np.array(observation0["edge_features"]["values"], dtype=np.float32), axis=-1))
                graph = BipartiteGraph(constraint_features, edge_indices, edge_features, variable_features, 
                                        candidates, candidate_choice, candidate_scores, info)
                rslt = database.add(graph)
                if not rslt:
                    logger("Skipping duplicate datapoint", results_loc)
                else:
                    logger("Saving to database", results_loc)

            else:
                count_no_improve += 1
                if count_no_improve >= 5:
                    break

            logger(f"Finished LNS step {s}: obj_val = {log_entry['primal_bound']}", results_loc)
        
        logger(f"Problem {instance_id}: best primal bound {LNS_log[-1]['primal_bound']}", results_loc)

        # out data storage
        out_queue.append({
            "type": "done",
            "episode": episode,
            "instance": instance,
            "seed": seed,
            "filename":filename,
            })
        
        model.freeProb()
        
    return out_queue

def send_orders(task, type, instances, seed, args, outdir):

    rng = np.random.RandomState(seed)
    episode = 0

    orders_queue = []

    for instance in instances:
        episode += 1
        seed = rng.randint(2**32)
        orders_queue.append([episode, task, type, instance, seed, args, outdir])

    return orders_queue

def collect_samples(task, type, instances, outdir, args):


    """
    Worker loop: fetch an instance, run an episode and record samples.
    """

    os.makedirs(outdir, exist_ok=True)
    print(outdir)
    rng = np.random.RandomState(args.seed + 1)

    # dir to keep samples temporarily; helps keep a prefect count
    tmp_samples_dir = f'{outdir}/'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    orders_queue = send_orders(task, type, instances, rng.randint(2**32), args, tmp_samples_dir)

    out_queue = make_samples(orders_queue)

    i = 0

    n_samples = len(instances)

    for i in range(n_samples):

        sample = out_queue[i]

        if sample['type'] == 'failed':
            i += 1

        if sample['type'] == 'done':
            filename = sample['filename']
            x = filename.split('/')[-1].split(".db")[0]
            print(x)
            print(outdir)
            # os.rename(filename, f"{outdir}/{x}.db")
            os.chmod(f"{outdir}/{x}.db", stat.S_IWUSR | stat.S_IRUSR)
            i+=1
            print(f"[m {os.getpid()}] {i} samples written, ep {sample['episode']}.")

