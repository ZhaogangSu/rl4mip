import argparse
import os
from multiprocessing import Process, Queue
import pickle
import numpy as np
import pyscipopt as scip
import sys
from rl4mip.DataCollector.lns_data.IL_data.utils import get_feat

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
def collect_feasols(filepath, settings):

    model = scip.Model()
    model.readProblem(filepath)

    model.setParam('limits/solutions', settings['maxsol'])
    model.setParam('limits/time', settings['maxtime'])
    model.setParam('parallel/maxnthreads', settings['threads'])

    model.optimize()

    sols = []
    objs = []

    mvars = model.getVars()

    oriVarNames = [var.name for var in mvars]
    
    sols = []
    objs = []
    solutions = model.getSols()  # Get the number of solutions

    for sol in solutions:
        # Get the solution values for all variables
        sols.append(np.array([model.getSolVal(sol, var) for var in mvars]))
        # Get the objective value of the solution
        objs.append(model.getSolObjVal(sol))

    sol_data = {
        'var_names':oriVarNames,
        'sols': sols,
        'objs': objs,
    }

    return sol_data

def collect(ins_dir, q, sol_dir, bg_dir, settings):
    
    while True:
        filename = q.get()
        if not filename:
            break

        filepath = os.path.join(ins_dir, filename)
        
        sol_data = collect_feasols(filepath, settings)
        
        pickle.dump(sol_data, open(os.path.join(sol_dir, filename+'.sol'), 'wb'))
        
        adj0, v_map0, v_nodes0, c_nodes0, b_vars0 = get_feat(filepath)
        
        BG_data0 = [adj0, v_map0, v_nodes0, c_nodes0, b_vars0]
        
        pickle.dump(BG_data0, open(os.path.join(bg_dir, filename+'.bg'),'wb'))

def collect_sols(task, problem, data_dir, dataset):


    Ins_Dir = os.path.join(data_dir, "instances", task, problem, dataset)
    SOL_DIR = os.path.join(data_dir, "sol_data", task, problem, dataset)
    BG_DIR = os.path.join(data_dir, "bg_data", task, problem, dataset)
    
    parent_dir = os.path.dirname(data_dir)
    # model_save_path = os.path.join(parent_dir, 'pre_models', task, problem)

    os.makedirs(SOL_DIR, exist_ok=True)
    os.makedirs(BG_DIR, exist_ok=True)
    # os.makedirs(model_save_path, exist_ok=True)

    filenames = os.listdir(Ins_Dir)

    nworkers = 5
    
    SETTINGS = {
        'maxtime': 360,
        'maxsol': 1000,
        'threads': 5,
        }

    q = Queue()

    for filename in filenames:
        q.put(filename)    

        # add stop signal
    N_workers = nworkers

        # gurobi settings

    for i in range(N_workers):
        q.put(None)     

    ps = []

    for i in range(N_workers):
        p = Process(target=collect,args=(Ins_Dir, q, SOL_DIR, BG_DIR, SETTINGS))
        p.start()
        ps.append(p)
        
    for p in ps:
        p.join()

    print('done')