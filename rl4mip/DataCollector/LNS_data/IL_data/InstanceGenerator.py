import os
import numpy as np
import sys
from .Generator_utils import Graph
from .Generator_utils import generate_setcover
from .Generator_utils import generate_indset
from .Generator_utils import generate_cauctions
from .Generator_utils import generate_maxcut

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def SetcoverGen(task, ntrain, nvalid, ntest_S, ntest_L, datapath, seed=0,
                nrows_S=2000, ncols_S=1000, nrow_L=2000, ncols_L=2000, dens=0.05, max_coef=100):
    """"""
    filenames = []
    nrowss = []
    ncolss = []
    denss = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/setcover/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nrowss.extend([nrows_S] * ntrain)
        ncolss.extend([ncols_S] * ntrain)
        denss.extend([dens] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/setcover/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nrowss.extend([nrows_S] * nvalid)
        ncolss.extend([ncols_S] * nvalid)
        denss.extend([dens] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/setcover/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nrowss.extend([nrows_S] * ntest_S)
        ncolss.extend([ncols_S] * ntest_S)
        denss.extend([dens] * ntest_S)

    if ntest_L:
        default_dir = f'instances/{task}/setcover/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nrowss.extend([nrow_L] * ntest_L)
        ncolss.extend([ncols_L] * ntest_L)
        denss.extend([dens] * ntest_L)

    # actually generate the instances
    for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
        print(f'  generating file {filename} ...')
        generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)

    return True

def CAGen(task, ntrain, nvalid, ntest_S, ntest_L, datapath, seed=0, 
          num_of_items_S=2000, num_of_bids_S=4000, num_of_items_L=4000, num_of_bids_L=8000):
    
    """"""
    filenames = []
    nitemss = []
    nbidss = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/cauctions/train'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(ntrain, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nitemss.extend([num_of_items_S] * ntrain)
        nbidss.extend([num_of_bids_S ] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/cauctions/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(nvalid, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nitemss.extend([num_of_items_S] * nvalid)
        nbidss.extend([num_of_bids_S ] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/cauctions/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(ntest_S, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nitemss.extend([num_of_items_S] * ntest_S)
        nbidss.extend([num_of_bids_S ] * ntest_S)

    if ntest_L:
        default_dir = f'instances/{task}/cauctions/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print("{} instances in {}".format(ntest_L, default_dir))
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nitemss.extend([num_of_items_L] * ntest_L)
        nbidss.extend([num_of_bids_L ] * ntest_L)

    # actually generate the instances
    for filename, nitems, nbids in zip(filenames, nitemss, nbidss):
        print(f'  generating file {filename} ...')
        generate_cauctions(rng, filename, n_items=nitems, n_bids=nbids, add_item_prob=0.7)

    print("done.")
    return True

def MaxCutGen(task, ntrain, nvalid, ntest_S, ntest_L, datapath, seed=0,
                number_of_nodes_S=1000, number_of_nodes_L=2000, ratio = 5):
    """"""
    filenames = []
    nnodess = []
    ratio_list = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/maxcut/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nnodess.extend([number_of_nodes_S] * ntrain)
        ratio_list.extend([ratio] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/maxcut/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nnodess.extend([number_of_nodes_S] * nvalid)
        ratio_list.extend([ratio] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/maxcut/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nnodess.extend([number_of_nodes_S] * ntest_S)
        ratio_list.extend([ratio] * ntest_S)

    if ntest_L:
        default_dir = f'instances/{task}/maxcut/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nnodess.extend([number_of_nodes_L] * ntest_L)
        ratio_list.extend([ratio] * ntest_L)

    for filename, ncs, nfs in zip(filenames, nnodess, ratio_list):
        print(f"  generating file {filename} ...")
        graph = Graph.barabasi_albert(ncs, nfs, rng)
        generate_maxcut(graph, filename)

    return True

def IndsetGen(task, ntrain, nvalid, ntest_S, ntest_L, datapath, seed=0,
                number_of_nodes_S=6000, number_of_nodes_L=12000, affinity=4):
    """"""
    filenames = []
    nnodess = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/indset/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nnodess.extend([number_of_nodes_S] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/indset/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nnodess.extend([number_of_nodes_S] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/indset/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nnodess.extend([number_of_nodes_S] * ntest_S)

    if ntest_L:
        default_dir = f'instances/{task}/indset/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nnodess.extend([number_of_nodes_L] * ntest_L)

    # actually generate the instances
    for filename, nnodes in zip(filenames, nnodess):    
        print(f"  generating file {filename} ...")
        graph = Graph.barabasi_albert(nnodes, affinity, rng)
        generate_indset(graph, filename)

    return True

def MVCGen(task, ntrain, nvalid, ntest_S, ntest_L, datapath, seed=0,
                number_of_nodes_S=6000, number_of_nodes_L=12000, affinity=4):
    """"""
    filenames = []
    nnodess = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/{task}/mvc/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntrain}.lp')])
        nnodess.extend([number_of_nodes_S] * ntrain)

    if nvalid:
        default_dir = f'instances/{task}/mvc/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{nvalid}.lp')])
        nnodess.extend([number_of_nodes_S] * nvalid)

    if ntest_S:
        default_dir = f'instances/{task}/mvc/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_S} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_S}.lp')])
        nnodess.extend([number_of_nodes_S] * ntest_S)

    if ntest_L:
        default_dir = f'instances/{task}/mvc/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest_L} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{ntest_L}.lp')])
        nnodess.extend([number_of_nodes_L] * ntest_L)

    # actually generate the instances
    for filename, nnodes in zip(filenames, nnodess):    
        print(f"  generating file {filename} ...")
        graph = Graph.barabasi_albert(nnodes, affinity, rng)
        generate_indset(graph, filename)

    return True


def InstancesGen_IL(task, problem='setcover', ntrain=10, nvalid=10, ntest_S=10, ntest_L=10, seed=0, datapath=None):
    
    if problem == 'setcover':
        SetcoverGen(task, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S, ntest_L=ntest_L, datapath=datapath, seed=seed, nrows_S=5000, ncols_S=4000, nrow_L=5000, ncols_L=8000)
        
    elif problem == 'indset':
        IndsetGen(task, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S, ntest_L=ntest_L, datapath=datapath, seed=seed, number_of_nodes_S=6000, number_of_nodes_L=12000)

    elif problem == 'mvc':
        MVCGen(task, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S, ntest_L=ntest_L, datapath=datapath, seed=seed, number_of_nodes_S=1000, number_of_nodes_L=2000)

    elif problem == 'cauctions':
        CAGen(task, ntrain=ntrain, nvalid=nvalid, ntest_S=ntest_S, ntest_L=ntest_L, datapath=datapath, seed=seed, num_of_items_S=2000, num_of_bids_S=4000, num_of_items_L=4000, num_of_bids_L=8000)

    else:
        raise NotImplementedError
