import os
import time
import datetime
import pickle
import pyscipopt as scip
import pandas as pd
import numpy as np

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
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)
            

def get_test_instances(datapath, problem_size, n_instance=10, shuffle=False):

    problem_folders = {
        'setcover_small': 'setcover/small_test',
        'setcover_medium': 'setcover/medium_test',
        'setcover_big': 'setcover/big_test',
        'cauctions_small': 'cauctions/small_test',
        'cauctions_medium': 'cauctions/medium_test',
        'cauctions_big': 'cauctions/big_test',
        'facilities_small': 'facilities/small_test',
        'facilities_medium': 'facilities/medium_test',
        'facilities_big': 'facilities/big_test',
        'indset_small': 'indset/small_test',
        'indset_medium': 'indset/medium_test',
        'indset_big': 'indset/big_test'
        }
    problem_folder = problem_folders[problem_size]

    instance_dir = os.path.join(datapath, 'instances', problem_folder)
    all_instance_names = os.listdir(instance_dir)

    if len(all_instance_names) >= n_instance:
        instance_names = np.random.choice(all_instance_names, size=n_instance, replace=False) if shuffle else all_instance_names[:n_instance]
    else:
        log(f"warning: exist instances {len(all_instance_names)} is less than required ({n_instance})")
        instance_names = all_instance_names

    instance_paths = [os.path.join(instance_dir, name) for name in instance_names]
    return instance_paths

def get_results_statistic(result_dict_list, file_name='', result_dir='', get_std=True, save_csv=False):

    result_dict_list.sort(key=lambda x: x["instance"])
    df = pd.DataFrame(result_dict_list)
    result_dict = df.mean(axis=0, numeric_only=True).to_dict()

    if get_std:
        std_result_dict = dict(df.std(axis=0, numeric_only=True))
        for k,v in std_result_dict.items():
            result_dict[f"std_{k}"] = v

    if save_csv:
        df.to_csv(os.path.join(result_dir, file_name))

    return result_dict, df

def _preprocess(state, mode='min-max-1'):
    state -= state.min(axis=0, keepdims=True)
    max_val = state.max(axis=0, keepdims=True)
    max_val[max_val == 0] = 1
    if mode=='min-max-1':
        state /= max_val
    elif mode=='min-max-2':
        state = 2 * state/max_val - 1
        # state[:,-1] = 1  # bias
    return state