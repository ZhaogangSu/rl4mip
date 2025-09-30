import os
import sys
import random
import numpy as np
import pyscipopt.scip as sp
from pathlib import Path 
from functools import partial
from .behaviour_utils import OracleNodeSelectorAbdel
from .behaviour_utils import LPFeatureRecorder, CompFeaturizer, CompFeaturizerSVM
from torch.multiprocessing import Process, set_start_method
from termcolor import colored
'''
Used for generating train dataset
'''

class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm):
        super().__init__(oracle_type)
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
        self.comp_behaviour_saver_svm = comp_behaviour_saver_svm
    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)

    def nodecomp(self, node1, node2):
        comp_res, comp_type = super().nodecomp(node1, node2, return_type=True)
        
        if comp_type in [-1,1]:
            self.comp_behaviour_saver.save_comp(self.model, 
                                                node1, 
                                                node2,
                                                comp_res,
                                                self.counter) 
            
            self.comp_behaviour_saver_svm.save_comp(self.model, 
                                                node1, 
                                                node2,
                                                comp_res,
                                                self.counter) 
        
            #print("saved comp # " + str(self.counter))
            self.counter += 1
        
        #make it bad to generate more data !
        if comp_type in [-1,1]:
            comp_res = -1 if comp_res == 1 else 1
        else:
            comp_res = 0
            
        return comp_res



def run_episode(oracle_type, instance,  save_dir, save_dir_svm, device):
    
    model = sp.Model()
    model.hideOutput()
    
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    # 使用并集参数，则需要关掉这些参数
    # # unable linear upgrading for constraint handler <logicor>
    # model.setParam('constraints/linear/upgrade/logicor',0)
    # # unable linear upgrading for constraint handler <indicator>
    # model.setParam('constraints/linear/upgrade/indicator',0)
    # # unable linear upgrading for constraint handler <knapsack>
    # model.setParam('constraints/linear/upgrade/knapsack', 0)
    # model.setParam('constraints/linear/upgrade/setppc', 0)
    # model.setParam('constraints/linear/upgrade/xor', 0)
    # model.setParam('constraints/linear/upgrade/varbound', 0)

    # 参数：5
    seed = 0
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
    model.setRealParam('limits/time', 3600)

    
    # ['setcover', 'indset', 'cauctions', 'facilities', 'maxcut'] 特殊处理
    found_items = [item for item in ['setcover', 'indset', 'cauctions', 'facilities', 'maxcut'] if item in instance]
    if len(found_items) == 1:
        optsol = model.readSolFile(instance.replace(".lp", ".sol").replace(found_items[0], found_items[0]+"_sol"))
    else:
        optsol = model.readSolFile(instance.replace(".lp", ".sol"))
    
    comp_behaviour_saver = CompFeaturizer(f"{save_dir}", instance_name=str(instance).split("/")[-1])
    # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    comp_behaviour_saver_svm = CompFeaturizerSVM(model, f"{save_dir_svm}", instance_name=str(instance).split("/")[-1])
    
    oracle_ns = OracleNodeSelRecorder(oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm)
    oracle_ns.setOptsol(optsol)
    oracle_ns.set_LP_feature_recorder(LPFeatureRecorder(model, device))
        
    
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing",
                         536870911,  536870911)


    # Run the optimizer
    model.optimize()
    print(f"Got behaviour for instance  "+ str(instance).split("/")[-1] + f' with {oracle_ns.counter} comparisons' )
    
    with open("nnodes.csv", "a+") as f:
        f.write(f"{model.getNNodes()},")
        f.close()
    with open("times.csv", "a+") as f:
        f.write(f"{model.getSolvingTime()},")
        f.close()
        
    return 1


def run_episodes(oracle_type, instances, save_dir, save_dir_svm, device):
    
    for instance in instances:
        run_episode(oracle_type, instance, save_dir, save_dir_svm, device)
        
    print("finished running episodes for process")
        
    return 1
    
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



def behaviour_generator(problem='GISP', ntrain=4, nvalid=4, ntest=4, n_cpu=16, datapath=None, 
                        oracle='optimal_plunger', data_partitions=['train', 'valid', 'test'], device = 'cpu'):
    try:
            set_start_method('spawn')
    except RuntimeError:
        ""
    
    with open("nnodes.csv", "w") as f:
        f.write("")
        f.close()
    with open("times.csv", "w") as f:
        f.write("")
        f.close()


    for data_partition in data_partitions:
        save_dir = os.path.join(datapath, f'behaviours/{problem}/{data_partition}')
        save_dir_svm = os.path.join(datapath, f'behaviours_svm/{problem}/{data_partition}')
        
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            ""
            
        try:
            os.makedirs(save_dir_svm)
        except FileExistsError:
            ""
        

        instances = list(Path(os.path.join(datapath, f"instances/{problem}/{data_partition}")).glob("*.lp"))
        if problem in ['setcover', 'indset', 'cauctions', 'facilities', 'maxcut']:
            instances = list(Path(os.path.join(datapath, f"instances/{problem+"_sol"}/{data_partition}")).glob("*.sol"))
            # for ins in instances:
            #     ins = str(ins).replace(problem+"_sol", problem).replace("sol", "lp")
            instances = [str(ins).replace(problem+"_sol", problem).replace(".sol", ".lp") for ins in instances]
            # print(instances[0])

        random.shuffle(instances)
        random.shuffle(instances)

        if data_partition == 'train' and len(instances) < ntrain:
            print(colored("The number of train data is less than ntrain.", "yellow"))
        elif data_partition == 'train':
            instances = instances[:ntrain]

        if data_partition == 'valid' and len(instances) < nvalid:
            print(colored("The number of valid data is less than nvalid.", "yellow"))
        elif data_partition == 'valid':
            instances = instances[:nvalid]

        if data_partition == 'test' and len(instances) < ntest:
            print(colored("The number of test data is less than ntest.", "yellow"))
        elif data_partition == 'test':
            instances = instances[:ntest]
    
        print(f"Generating {data_partition} samples from {len(instances)} instances using oracle {oracle}")

        processes = [ Process(name=f"worker {p}", 
                                        target=partial(run_episodes,
                                                        oracle_type=oracle,
                                                        instances=instances[ p1 : p2], 
                                                        save_dir=save_dir,
                                                        save_dir_svm=save_dir_svm,
                                                        device=device))
                        for p,(p1,p2) in enumerate(distribute(len(instances), n_cpu))]
        
        # set_start_method('spawn') 方法需要放在 "if __name__ == '__main__':" 下
        try:
            set_start_method('spawn')
        except RuntimeError:
            ""
            
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes


    nnodes = np.genfromtxt("nnodes.csv", delimiter=",")[:-1]
    times = np.genfromtxt("times.csv", delimiter=",")[:-1]

    print(f"Mean number of node created  {np.mean(nnodes)}")
    print(f"Mean solving time  {np.mean(times)}")
    print(f"Median number of node created  {np.median(nnodes)}")
    print(f"Median solving time  {np.median(times)}")
        