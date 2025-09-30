import os
import sys
from rl4mip.DataCollector.LNS_data.CL_data.Collector_utils import collect_samples

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Args:
    def __init__(self, problem, neighborhood):
        self.seed = 0
        self.problem = problem
        self.njobs = 1
        self.init_time_limit = 10
        self.neighborhood_size = neighborhood
        self.num_solve_steps = 30
        self.collect_time_limit = 3600
        self.sub_time_limit = 3600

def IMCollector(neighborhood, task, problem, ntrain, nvalid, datapath):

    args = Args(problem, neighborhood)
    
    """"""

    instances_train = [os.path.join(datapath, f'instances/{task}/{problem}/train/instance_{ntrain}.lp')]
    instances_valid = [os.path.join(datapath, f'instances/{task}/{problem}/valid/instance_{nvalid}.lp')]
    # instances_test = [os.path.join(datapath, f'instances/{task}/{problem}/valid/instance_{ntest}.lp')]
    
    out_dir = os.path.join(datapath, f'samples/{task}/{problem}')

    print(f"{len(instances_train)} train instances for {ntrain} samples")
    print(f"{len(instances_valid)} valid instances for {nvalid} samples")

    if ntrain == 0:
        print("Failed: 0 train data collection")
    else:
        collect_samples(task, 'train', instances_train, out_dir +"/train", args)
        print("Success: Train data collection")

    if nvalid == 0:
        print("Falied: 0 valid data collection")
    else:
        collect_samples(task, 'valid', instances_valid, out_dir +"/valid", args)
        print("Success: Valid data collection")

    # if ntest == 0:
    #     print("Falied: 0 test data collection")
    # else:
    #     collect_samples(task, 'test', instances_test, out_dir +"/test", args)
    #     print("Success: Test data collection")

    return True