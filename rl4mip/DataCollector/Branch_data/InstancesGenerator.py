import os
import numpy as np
from .Generator_utils import Graph
from .Generator_utils import generate_setcover
from .Generator_utils import generate_indset
from .Generator_utils import generate_cauctions
from .Generator_utils import generate_capacited_facility_location

def SetcoverGen(ntrain, nvalid, ntest, datapath, seed=0):
    """"""
    nrows = 500
    ncols = 1000
    dens = 0.05
    max_coef = 100

    filenames = []
    nrowss = []
    ncolss = []
    denss = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/setcover/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntrain)])
        nrowss.extend([nrows] * ntrain)
        ncolss.extend([ncols] * ntrain)
        denss.extend([dens] * ntrain)

    if nvalid:
        default_dir = f'instances/setcover/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(nvalid)])
        nrowss.extend([nrows] * nvalid)
        ncolss.extend([ncols] * nvalid)
        denss.extend([dens] * nvalid)

    if ntest:
        # default_dir = f'instances/setcover/test'
        # lp_dir = os.path.join(datapath, default_dir)
        # print(f"{ntest} instances in {lp_dir}")
        # os.makedirs(lp_dir, exist_ok=True)
        # filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        # nrowss.extend([nrows] * ntest)
        # ncolss.extend([ncols] * ntest)
        # denss.extend([dens] * ntest)

        nrows = 500
        default_dir = f'instances/setcover/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nrowss.extend([nrows] * ntest)
        ncolss.extend([ncols] * ntest)
        denss.extend([dens] * ntest)

        nrows = 1000
        default_dir = f'instances/setcover/medium_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nrowss.extend([nrows] * ntest)
        ncolss.extend([ncols] * ntest)
        denss.extend([dens] * ntest)

        nrows = 2000
        default_dir = f'instances/setcover/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nrowss.extend([nrows] * ntest)
        ncolss.extend([ncols] * ntest)
        denss.extend([dens] * ntest)

    # actually generate the instances
    for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
        print(f'  generating file {filename} ...')
        generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)

    return True

def IndsetGen(ntrain, nvalid, ntest, datapath, seed=0):
    """"""

    number_of_nodes = 500
    affinity = 4

    filenames = []
    nnodess = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/indset/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntrain)])
        nnodess.extend([number_of_nodes] * ntrain)

    if nvalid:
        default_dir = f'instances/indset/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(nvalid)])
        nnodess.extend([number_of_nodes] * nvalid)

    if ntest:
        # default_dir = f'instances/indset/test'
        # lp_dir = os.path.join(datapath, default_dir)
        # print(f"{ntest} instances in {lp_dir}")
        # os.makedirs(lp_dir, exist_ok=True)
        # filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        # nnodess.extend([number_of_nodes] * ntest)

        number_of_nodes = 500
        default_dir = f'instances/indset/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nnodess.extend([number_of_nodes] * ntest)

        number_of_nodes = 1000
        default_dir = f'instances/indset/medium_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nnodess.extend([number_of_nodes] * ntest)

        number_of_nodes = 1500
        default_dir = f'instances/indset/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nnodess.extend([number_of_nodes] * ntest)

    # actually generate the instances
    for filename, nnodes in zip(filenames, nnodess):
        print(f"  generating file {filename} ...")
        graph = Graph.barabasi_albert(nnodes, affinity, rng)
        generate_indset(graph, filename)

    return True

def CauctionsGen(ntrain, nvalid, ntest, datapath, seed=0):

    number_of_items=100
    number_of_bids=500

    filenames = []
    nitemss = []
    nbidss = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/cauctions/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntrain)])
        nitemss.extend([number_of_items] * ntrain)
        nbidss.extend([number_of_bids ] * ntrain)

    if nvalid:
        default_dir = f'instances/cauctions/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(nvalid)])
        nitemss.extend([number_of_items] * nvalid)
        nbidss.extend([number_of_bids ] * nvalid)

    if ntest:
        # default_dir = f'instances/cauctions/test'
        # lp_dir = os.path.join(datapath, default_dir)
        # print(f"{ntest} instances in {lp_dir}")
        # os.makedirs(lp_dir, exist_ok=True)
        # filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        # nitemss.extend([number_of_items] * ntest)
        # nbidss.extend([number_of_bids ] * ntest)

        number_of_items = 100
        number_of_bids = 500
        default_dir = f'instances/cauctions/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nitemss.extend([number_of_items] * ntest)
        nbidss.extend([number_of_bids ] * ntest)

        number_of_items = 200
        number_of_bids = 1000
        default_dir = f'instances/cauctions/medium_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nitemss.extend([number_of_items] * ntest)
        nbidss.extend([number_of_bids ] * ntest)

        number_of_items = 300
        number_of_bids = 1500
        default_dir = f'instances/cauctions/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        nitemss.extend([number_of_items] * ntest)
        nbidss.extend([number_of_bids ] * ntest)

    # actually generate the instances
    for filename, nitems, nbids in zip(filenames, nitemss, nbidss):
        print(f"  generating file {filename} ...")
        generate_cauctions(rng, filename, n_items=nitems, n_bids=nbids, add_item_prob=0.7)

def FacilitiesGen(ntrain, nvalid, ntest, datapath, seed=0):
    
    number_of_customers=100
    number_of_facilities=100
    ratio = 5

    filenames = []
    ncustomerss = []
    nfacilitiess = []
    ratios = []
    rng = np.random.RandomState(seed)

    if ntrain:
        default_dir = f'instances/facilities/train'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntrain} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntrain)])
        ncustomerss.extend([number_of_customers] * ntrain)
        nfacilitiess.extend([number_of_facilities] * ntrain)
        ratios.extend([ratio] * ntrain)

    if nvalid:
        default_dir = f'instances/facilities/valid'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{nvalid} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(nvalid)])
        ncustomerss.extend([number_of_customers] * nvalid)
        nfacilitiess.extend([number_of_facilities] * nvalid)
        ratios.extend([ratio] * nvalid)

    if ntest:
        # default_dir = f'instances/facilities/test'
        # lp_dir = os.path.join(datapath, default_dir)
        # print(f"{ntest} instances in {lp_dir}")
        # os.makedirs(lp_dir, exist_ok=True)
        # filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        # ncustomerss.extend([number_of_customers] * ntest)
        # nfacilitiess.extend([number_of_facilities] * ntest)
        # ratios.extend([ratio] * ntest)

        number_of_customers = 100
        number_of_facilities = 100
        default_dir = f'instances/facilities/small_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        ncustomerss.extend([number_of_customers] * ntest)
        nfacilitiess.extend([number_of_facilities] * ntest)
        ratios.extend([ratio] * ntest)

        number_of_customers = 200
        default_dir = f'instances/facilities/medium_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        ncustomerss.extend([number_of_customers] * ntest)
        nfacilitiess.extend([number_of_facilities] * ntest)
        ratios.extend([ratio] * ntest)

        number_of_customers = 400
        default_dir = f'instances/facilities/big_test'
        lp_dir = os.path.join(datapath, default_dir)
        print(f"{ntest} instances in {lp_dir}")
        os.makedirs(lp_dir, exist_ok=True)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(ntest)])
        ncustomerss.extend([number_of_customers] * ntest)
        nfacilitiess.extend([number_of_facilities] * ntest)
        ratios.extend([ratio] * ntest)

    # actually generate the instances
    for filename, ncs, nfs, r in zip(filenames, ncustomerss, nfacilitiess, ratios):
        print(f"  generating file {filename} ...")
        generate_capacited_facility_location(rng, filename, n_customers=ncs, n_facilities=nfs, ratio=r)

def InstancesGen(problem='setcover', ntrain=10, nvalid=10, ntest=10, seed=0, datapath=None):

    if problem == 'setcover':
        SetcoverGen(ntrain=ntrain, nvalid=nvalid, ntest=ntest, datapath=datapath, seed=seed)
        
    elif problem == 'indset':
        IndsetGen(ntrain=ntrain, nvalid=nvalid, ntest=ntest, datapath=datapath, seed=seed)

    elif problem == 'cauctions':
        CauctionsGen(ntrain=ntrain, nvalid=nvalid, ntest=ntest, datapath=datapath, seed=seed)

    elif problem == 'facilities':
        FacilitiesGen(ntrain=ntrain, nvalid=nvalid, ntest=ntest, datapath=datapath, seed=seed)

    else:
        raise NotImplementedError