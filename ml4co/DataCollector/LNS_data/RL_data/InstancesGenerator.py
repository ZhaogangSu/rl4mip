import os
import numpy as np
from .Generator_utils import Graph
from .Generator_utils import generate_setcover
from .Generator_utils import generate_indset
from .Generator_utils import generate_cauctions
from .Generator_utils import generate_maxcut

def SetcoverGen(task, problem, ntrain, nvalid, ntest_S, ntest_M, ntest_L, seed=0, datapath=None, 
                nrows=5000, ncols=1000, dens=0.05, max_coef=100):
    """"""
    filenames = []
    nrowss = []
    ncolss = []
    denss = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/{problem}/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nrowss.extend([nrows] * ntrain)
        ncolss.extend([ncols] * ntrain)
        denss.extend([dens] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/{problem}/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nrowss.extend([nrows] * nvalid)
        ncolss.extend([ncols] * nvalid)
        denss.extend([dens] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/{problem}/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nrowss.extend([nrows] * ntest_S)
        ncolss.extend([ncols] * ntest_S)
        denss.extend([dens] * ntest_S)

    if ntest_M:
        ncols = 2000
        default_dir = f'instances/{task}/{problem}/medium_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_M} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_M}.lp')])
        nrowss.extend([nrows] * ntest_M)
        ncolss.extend([ncols] * ntest_M)
        denss.extend([dens] * ntest_M)

    if ntest_L:
        ncols = 4000
        default_dir = f'instances/{task}/{problem}/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nrowss.extend([nrows] * ntest_L)
        ncolss.extend([ncols] * ntest_L)
        denss.extend([dens] * ntest_L)

    # actually generate the instances
    for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
        print(f'  generating file {filename} ...')
        generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)

    return True

def CAGen(task, problem, ntrain, nvalid, ntest_S, ntest_M, ntest_L, seed=0, datapath=None, 
          num_of_items = 2000, num_of_bids=4000):
    """"""
    filenames = []
    nitemss = []
    nbidss = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/{problem}/train'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(ntrain, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nitemss.extend([num_of_items] * ntrain)
        nbidss.extend([num_of_bids ] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/{problem}/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(nvalid, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nitemss.extend([num_of_items] * nvalid)
        nbidss.extend([num_of_bids ] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/{problem}/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(ntest_S, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nitemss.extend([num_of_items] * ntest_S)
        nbidss.extend([num_of_bids ] * ntest_S)

    if ntest_M:
        num_of_items = 4000
        num_of_bids = 8000
        default_dir = f'instances/{task}/{problem}/medium_test'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(ntest_M, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_M}.lp')])
        nitemss.extend([num_of_items] * ntest_M)
        nbidss.extend([num_of_bids ] * ntest_M)
    
    if ntest_L:
        num_of_items = 8000
        num_of_bids = 16000
        default_dir = f'instances/{task}/{problem}/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(ntest_L, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nitemss.extend([num_of_items] * ntest_L)
        nbidss.extend([num_of_bids ] * ntest_L)

    # actually generate the instances
    for filename, nitems, nbids in zip(filenames, nitemss, nbidss):
        generate_cauctions(rng, filename, n_items=nitems, n_bids=nbids, add_item_prob=0.7)

    print("done.")
    return True

def MaxCutGen(task, problem, ntrain, nvalid, ntest_S, ntest_M, ntest_L, seed=0, datapath=None, 
                number_of_nodes=500, ratio = 5):
    """"""
    filenames = []
    nnodess = []
    ratio_list = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/{problem}/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nnodess.extend([number_of_nodes] * ntrain)
        ratio_list.extend([ratio] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/{problem}/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nnodess.extend([number_of_nodes] * nvalid)
        ratio_list.extend([ratio] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/{problem}/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nnodess.extend([number_of_nodes] * ntest_S)
        ratio_list.extend([ratio] * ntest_S)

    if ntest_M:
        number_of_nodes = 1000
        default_dir = f'instances/{task}/{problem}/medium_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_M} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_M}.lp')])
        nnodess.extend([number_of_nodes] * ntest_M)
        ratio_list.extend([ratio] * ntest_M)

    if ntest_L:
        number_of_nodes = 2000
        default_dir = f'instances/{task}/{problem}/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nnodess.extend([number_of_nodes] * ntest_L)
        ratio_list.extend([ratio] * ntest_L)

    for filename, ncs, nfs in zip(filenames, nnodess, ratio_list):
        print(f"  generating file {filename} ...")
        graph = Graph.barabasi_albert(ncs, nfs, rng)
        generate_maxcut(graph, filename)

    return True

def IndsetGen(task, problem, ntrain, nvalid, ntest_S, ntest_M, ntest_L, seed=0, datapath=None,
                number_of_nodes=1500, affinity=4):
    """"""
    filenames = []
    nnodess = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/{problem}/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nnodess.extend([number_of_nodes] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/{problem}/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nnodess.extend([number_of_nodes] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/{problem}/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nnodess.extend([number_of_nodes] * ntest_S)

    if ntest_M:
        number_of_nodes = 3000
        default_dir = f'instances/{task}/{problem}/medium_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_M} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_M}.lp')])
        nnodess.extend([number_of_nodes] * ntest_M)

    if ntest_L:
        number_of_nodes = 6000
        default_dir = f'instances/{task}/{problem}/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nnodess.extend([number_of_nodes] * ntest_L)

    # actually generate the instances
    for filename, nnodes in zip(filenames, nnodess):    
        print(f"  generating file {filename} ...")
        graph = Graph.barabasi_albert(nnodes, affinity, rng)
        generate_indset(graph, filename)

    return True

def InstancesGen_RL(task, problem='setcover', ntrain=10, nvalid=10, ntest_S=10, ntest_M=10, ntest_L=10, seed=0, datapath=None):
    
    if problem == 'setcover':
        SetcoverGen(task, problem, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S,  ntest_M=ntest_M, ntest_L=ntest_L, seed=seed, datapath=datapath, nrows=5000, ncols=1000, dens=0.05, max_coef=100) # 4000, 5000
        
    elif problem == 'indset':
        IndsetGen(task, problem, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S,  ntest_M=ntest_M, ntest_L=ntest_L, seed=seed, datapath=datapath, number_of_nodes=1500, affinity=4) # 6000

    elif problem == 'cauctions':
        CAGen(task, problem, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S,  ntest_M=ntest_M, ntest_L=ntest_L, seed=seed, datapath=datapath, num_of_items = 2000, num_of_bids=4000) # 16000 8000

    elif problem == 'maxcut':
        MaxCutGen(task, problem, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S,  ntest_M=ntest_M, ntest_L=ntest_L, seed=seed, datapath=datapath, number_of_nodes=500, ratio = 5) # 11975

    else:
        raise NotImplementedError