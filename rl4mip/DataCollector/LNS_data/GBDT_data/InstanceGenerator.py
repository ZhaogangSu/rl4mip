import os
import pickle
import random
from .Generator_utils import write_lp_file_Max, write_lp_file_Min
from .Generator_utils import generate_SC
from .Generator_utils import generate_IS
from .Generator_utils import generate_CA
from .Generator_utils import generate_MVC

def SetcoverGen(task, problem, ntrain, nvalid, ntest_S, ntest_M, ntest_L, seed=0, datapath=None, N=20000, M=20000):
    
    random.seed(seed) 

    if ntrain:

        default_dir = f'instances/{task}/pickle_data/{problem}/train'
        default_lp = f'instances/{task}/lp_data/{problem}/train'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {pickle_dir}")


        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_SC(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntrain) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, lower_bound, upper_bound, value_type, coefficient], f)

        lp_dir = os.path.join(datapath, default_lp)
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntrain)

    if nvalid:

        default_dir = f'instances/{task}/pickle_data/{problem}/valid'
        default_lp = f'instances/{task}/lp_data/{problem}/valid'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_SC(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(nvalid) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, nvalid)

    if ntest_S:

        default_dir = f'instances/{task}/pickle_data/{problem}/small_test'
        default_lp = f'instances/{task}/lp_data/{problem}/small_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_SC(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_S) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_S)
    
    if ntest_M:
        
        N = 200000
        M = 200000

        default_dir = f'instances/{task}/pickle_data/{problem}/medium_test'
        default_lp = f'instances/{task}/lp_data/{problem}/medium_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_M} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_IS(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_M) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_M)

    if ntest_L:
        
        N = 2000000
        M = 2000000

        default_dir = f'instances/{task}/pickle_data/{problem}/big_test'
        default_lp = f'instances/{task}/lp_data/{problem}/big_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {pickle_dir}")


        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_IS(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_L) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_L)
    
    return True

def MVCGen(task, problem, ntrain, nvalid, ntest_S, ntest_M, ntest_L, seed=0, datapath=None, N=10000, M=30000):
    
    random.seed(seed) 

    if ntrain:

        default_dir = f'instances/{task}/pickle_data/{problem}/train'
        default_lp = f'instances/{task}/lp_data/{problem}/train'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_MVC(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntrain) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntrain)

    if nvalid:

        default_dir = f'instances/{task}/pickle_data/{problem}/valid'
        default_lp = f'instances/{task}/lp_data/{problem}/valid'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {pickle_dir}")


        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_MVC(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(nvalid) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, nvalid)

    if ntest_S:

        default_dir = f'instances/{task}/pickle_data/{problem}/small_test'
        default_lp = f'instances/{task}/lp_data/{problem}/small_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_MVC(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_S) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_S)
    
    if ntest_M:
        
        N = 100000
        M = 300000

        default_dir = f'instances/{task}/pickle_data/{problem}/medium_test'
        default_lp = f'instances/{task}/lp_data/{problem}/medium_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_M} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_MVC(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_M) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_M)

    if ntest_L:
        
        N = 1000000
        M = 3000000

        default_dir = f'instances/{task}/pickle_data/{problem}/big_test'
        default_lp = f'instances/{task}/lp_data/{problem}/big_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_MVC(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_L) + '.pickle', 'wb') as f:
            pickle.dump(['minimize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_L)
    
    return True

def IndsetGen(task, problem, ntrain, nvalid, ntest_S, ntest_M, ntest_L, seed=0, datapath=None, N=10000, M=30000):
    
    random.seed(seed) 

    if ntrain:

        default_dir = f'instances/{task}/pickle_data/{problem}/train'
        default_lp = f'instances/{task}/lp_data/{problem}/train'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_IS(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntrain) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntrain)

    if nvalid:

        default_dir = f'instances/{task}/pickle_data/{problem}/valid'
        default_lp = f'instances/{task}/lp_data/{problem}/valid'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_IS(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(nvalid) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, nvalid)

    if ntest_S:

        default_dir = f'instances/{task}/pickle_data/{problem}/small_test'
        default_lp = f'instances/{task}/lp_data/{problem}/small_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_IS(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_S) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_S)
    
    if ntest_M:
        
        N = 100000
        M = 300000

        default_dir = f'instances/{task}/pickle_data/{problem}/medium_test'
        default_lp = f'instances/{task}/lp_data/{problem}/medium_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_M} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_IS(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_M) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_M)

    if ntest_L:
        
        N = 1000000
        M = 3000000

        default_dir = f'instances/{task}/pickle_data/{problem}/big_test'
        default_lp = f'instances/{task}/lp_data/{problem}/big_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {pickle_dir}")


        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_IS(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_L) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_L)
            
    return True

def CAGen(task, problem, ntrain, nvalid, ntest_S, ntest_M, ntest_L, seed=0, datapath=None, N=10000, M=30000):
    
    random.seed(seed) 

    if ntrain:

        default_dir = f'instances/{task}/pickle_data/{problem}/train'
        default_lp = f'instances/{task}/lp_data/{problem}/train'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_CA(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntrain) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntrain)

    if nvalid:

        default_dir = f'instances/{task}/pickle_data/{problem}/valid'
        default_lp = f'instances/{task}/lp_data/{problem}/valid'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_CA(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(nvalid) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, nvalid)

    if ntest_S:

        default_dir = f'instances/{task}/pickle_data/{problem}/small_test'
        default_lp = f'instances/{task}/lp_data/{problem}/small_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_CA(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_S) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_S)
    
    if ntest_M:
        
        N = 100000
        M = 300000

        default_dir = f'instances/{task}/pickle_data/{problem}/medium_test'
        default_lp = f'instances/{task}/lp_data/{problem}/medium_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_M} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_CA(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_M) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_M)

    if ntest_L:
        
        N = 1000000
        M = 3000000

        default_dir = f'instances/{task}/pickle_data/{problem}/big_test'
        default_lp = f'instances/{task}/lp_data/{problem}/big_test'
        
        pickle_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {pickle_dir}")

        n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type = generate_CA(N, M)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        with open(f'{pickle_dir}/instance_' + str(ntest_L) + '.pickle', 'wb') as f:
            pickle.dump(['maximize', n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type], f)

        lp_dir = os.path.join(datapath, default_lp) 
        if not os.path.exists(lp_dir): 
            os.makedirs(lp_dir, exist_ok=True)
        write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, ntest_L)
    
    return True

def InstancesGen_GBDT(task, problem='setcover', ntrain=10, nvalid=10, ntest_S=10, ntest_M=10, ntest_L=10, seed=0, datapath=None):
    
    if problem == 'setcover':
        SetcoverGen(task, problem=problem, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S, ntest_M=ntest_M, ntest_L=ntest_L, seed=seed, datapath=datapath, N=20000, M=20000)
        
    elif problem == 'indset':
        IndsetGen(task, problem=problem, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S, ntest_M=ntest_M, ntest_L=ntest_L, seed=seed, datapath=datapath, N=10000, M=30000)

    elif problem == 'mvc':
        MVCGen(task, problem=problem, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S, ntest_M=ntest_M, ntest_L=ntest_L, seed=seed, datapath=datapath, N=10000, M=30000)

    elif problem == 'cauctions':
        CAGen(task, problem=problem, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S, ntest_M=ntest_M, ntest_L=ntest_L, seed=seed, datapath=datapath, N=10000, M=30000)
    else:
        raise NotImplementedError