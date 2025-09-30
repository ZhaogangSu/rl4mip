import os
import glob
import numpy as np
from .Colletor_utils import collect_samples

def FSBCollector(problem, train_size, valid_size, test_size, datapath,
                 time_limit=3600, seed=0, njobs=1):
    """"""
    if problem == 'setcover':
        instances_train = glob.glob(os.path.join(datapath, 'instances/setcover/train/*.lp'))
        instances_valid = glob.glob(os.path.join(datapath, 'instances/setcover/valid/*.lp'))
        instances_test = glob.glob(os.path.join(datapath, 'instances/setcover/test/*.lp'))
        out_dir = os.path.join(datapath, 'samples/setcover')

    elif problem == 'indset':
        instances_train = glob.glob(os.path.join(datapath, 'instances/indset/train/*.lp'))
        instances_valid = glob.glob(os.path.join(datapath, 'instances/indset/valid/*.lp'))
        instances_test = glob.glob(os.path.join(datapath, 'instances/indset/test/*.lp'))
        out_dir = os.path.join(datapath, 'samples/indset')
    
    elif problem == 'cauctions':
        instances_train = glob.glob(os.path.join(datapath, 'instances/cauctions/train/*.lp'))
        instances_valid = glob.glob(os.path.join(datapath, 'instances/cauctions/valid/*.lp'))
        instances_test = glob.glob(os.path.join(datapath, 'instances/cauctions/test/*.lp'))
        out_dir = os.path.join(datapath, 'samples/cauctions')

    elif problem == 'facilities':
        instances_train = glob.glob(os.path.join(datapath, 'instances/facilities/train/*.lp'))
        instances_valid = glob.glob(os.path.join(datapath, 'instances/facilities/valid/*.lp'))
        instances_test = glob.glob(os.path.join(datapath, 'instances/facilities/test/*.lp'))
        out_dir = os.path.join(datapath,'samples/facilities')

    else:
        raise NotImplementedError
    
    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")


    rng = np.random.RandomState(seed + 1)
    if len(instances_train) > 0 and train_size > 0:
        collect_samples(instances_train, out_dir +"/train", rng, train_size, njobs, time_limit)
        print("Success: Train data collection")
    
    if len(instances_valid) > 0 and valid_size > 0:
        collect_samples(instances_valid, out_dir +"/valid", rng, valid_size, njobs, time_limit)
        print("Success: Valid data collection")

    if len(instances_test) > 0 and test_size > 0:
        collect_samples(instances_test, out_dir +"/test", rng, test_size, njobs, time_limit)
        print("Success: Test data collection")

    return True


