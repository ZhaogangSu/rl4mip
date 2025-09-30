import os
import numpy as np
import torch
import shutil
import pyscipopt as scip
import time
import shutil
from ml4co.DataCollector.LNS_data.RL_data import utilities
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
def make_samples(in_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    episode, instance, obs, actions, seed, exploration_policy, eval_flag, time_limit, out_dir = in_queue
    print('[w {}] episode {}, seed {}, processing instance \'{}\'...'.format(os.getpid(),episode,seed,instance)) 

    if eval_flag==1:
        seed=0
    else:
        seed=0
  
    m = scip.Model()
    m.setIntParam('display/verblevel', 5)
    m.readProblem('{}'.format(instance))
    utilities.init_scip_params(m, seed=seed)
    m.setIntParam('timing/clocktype', 1)
    m.setRealParam('limits/time', time_limit) 

        
    varss = [x for x in m.getVars()]             
    
    if eval_flag==1:

        minimum_k = np.where(np.array(actions.squeeze())<0.5)
        max_k = np.where(np.array(actions.squeeze())>0.5)[0]
        min_k = minimum_k
        for i in min_k[0]:
            a,b = m.fixVar(varss[i],obs[i])        
    else:
#        minimum_k = np.where(np.random.binomial(1, actions) <0.5)
        minimum_k = np.where(np.array(actions.squeeze())<0.5)
        max_k = np.where(np.array(actions.squeeze())>0.5)[0]
        min_k = minimum_k[0]
        for i in min_k:
            a,b = m.fixVar(varss[i],obs[i])     
   
    m.optimize()
    
    K = [m.getVal(x) for x in m.getVars()]  
        
    obj = m.getObjVal()

    print(m.getStatus())

    m.freeProb()    

    out_queue = {
        'type': 'solution',
        'episode': episode,
        'instance': instance,
        'sol' : np.array(K),
        'obj' : obj,
        'seed': seed,
        'mask': max_k,
    }            

    print("[w {}] episode {} done".format(os.getpid(),episode))
    
    return out_queue


def send_orders(instances, epi, obs, actions, seed, exploration_policy, eval_flag, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

#    episode = 0
    orders_queue = []
    for i in range(len(instances)):
#        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.append([epi[i], instances[i], obs[i], actions[i], seed, exploration_policy, eval_flag, time_limit, out_dir])
#        episode += 1

    return orders_queue


def collect_samples(instances, epi, obs, actions, out_dir, rng, n_samples,
                    exploration_policy, eval_flag, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """

    tmp_samples_dir = '{}/tmp'.format(out_dir)
    # os.makedirs(tmp_samples_dir, exist_ok=True)


    pars = send_orders(instances, epi, obs, actions, rng.randint(2**32), exploration_policy, eval_flag, time_limit, tmp_samples_dir) 
    
    out_Q = []
    for n in range(n_samples):
        out_queue = make_samples(pars[n])
        out_Q.append(out_queue)        


    # record answers 
    i = 0
    collecter=[]
    epi=[]
    obje=[]
    instances=[]
    mask=[]

    for sample in out_Q:
        
        collecter.append(sample['sol'])
        
        epi.append(sample['episode'])
        
        obje.append(sample['obj'])

        instances.append(sample['instance'])

        mask.append(sample['mask'])
        
        i += 1

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)
   
    return np.stack(collecter), np.stack(epi), np.stack(obje), instances, mask
    
class SamplingAgent0(scip.Branchrule):

    def __init__(self, episode, instance, seed, exploration_policy, out_dir):
        self.episode = episode
        self.instance = instance
        self.seed = seed
        self.exploration_policy = exploration_policy
        self.out_dir = out_dir

        self.rng = np.random.RandomState(seed)
        self.new_node = True
        self.sample_counter = 0

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}

    def branchexeclp(self, allowaddcons):

        # custom policy branching           
        if self.model.getNNodes() == 1:    
            
            # extract formula features
            self.state = utilities.extract_state(self.model, self.state_buffer)              

            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)
                               
        elif self.model.getNNodes() != 1:
               
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)   
            
        else:
            raise NotImplementedError

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}

def make_samples0(in_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """
    episode, instance, seed, exploration_policy, eval_flag, time_limit, out_dir = in_queue
    print('[w {}] episode {}, seed {}, processing instance \'{}\'...'.format(os.getpid(),episode,seed,instance))

    if eval_flag==1:
        seed=0
    else:
        seed=0
    
    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    # m.setIntParam('limits/time', time_limit)
    m.readProblem('{}'.format(instance))
    utilities.init_scip_paramsH(m, seed=seed)
    m.setIntParam('timing/clocktype', 2)
    m.setLongintParam('limits/nodes', 1) 

    branchrule = SamplingAgent0(
        episode=episode,
        instance=instance,
        seed=seed,
        exploration_policy=exploration_policy,
        out_dir=out_dir)

    m.includeBranchrule(
        branchrule=branchrule,
        name="Sampling branching rule", desc="",
        priority=666666, maxdepth=-1, maxbounddist=1)
    
    abc=time.time()    
    m.optimize()       
    print(time.time()-abc)    

    b_obj = m.getObjVal()

    K = [m.getVal(x) for x in m.getVars()]     

    out_queue = {
        'type': 'formula',
        'episode': episode,
        'instance': instance,
        'state' : branchrule.state,
        'seed': seed,
        'b_obj': b_obj,
        'sol' : np.array(K),        
    }  
       
    m.freeTransform()  
        
    obj = [x.getObj() for x in m.getVars()]  
    
    out_queue['obj_coeff'] = obj
    
    m.freeProb()  
        
    print("[w {}] episode {} done".format(os.getpid(),episode))
    
    return out_queue


def send_orders0(instances, n_samples, seed, exploration_policy, batch_id, eval_flag, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    st = batch_id*n_samples
    orders_queue = []
    for i in instances[st:st+n_samples]:     
        seed = rng.randint(2**32)
        orders_queue.append([episode, i, seed, exploration_policy, eval_flag, time_limit, out_dir])
        episode += 1

    return orders_queue

def collect_samples0(instances, out_dir, rng, n_samples,
                    exploration_policy, batch_id, eval_flag, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    # os.makedirs(out_dir, exist_ok=True)

    tmp_samples_dir = '{}/tmp'.format(out_dir)
    # os.makedirs(tmp_samples_dir, exist_ok=True)
    
    pars = send_orders0(instances, n_samples, rng.randint(2**32), exploration_policy, batch_id, eval_flag, time_limit, tmp_samples_dir)  
    
    out_Q = []
    for n in range(n_samples):
        out_queue = make_samples0(pars[n])
        out_Q.append(out_queue)        
        
    i = 0
    feature_graph = []
    # collecterM=[]

    epi=[]
    instances=[]
    # obje=[]
    bobj=[]
    ini_sol=[]
    # feats_graph = []
    
    for sample in out_Q:

        collecter={}
        ini_sol.append(sample['sol']) # 初始解         

        X=np.stack(sample['state'][1]['values']).reshape(-1,19)
        X = np.where(X == None, np.nan, X).astype(float)
        feats = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18]]
        min_vals = np.nanmin(feats, axis=0)
        max_vals = np.nanmax(feats, axis=0)

        # 避免除以 0（当 max = min）
        denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)

        X = (feats - min_vals) / denom

        collecter['variable_features'] = torch.tensor(X, dtype=torch.float32) 
        collecter['constraint_features'] = torch.tensor(sample['state'][0]['values'], dtype=torch.float32)
        collecter['edge_indices'] = sample['state'][2].float() # 
        collecter['edge_features'] = torch.tensor(sample['state'][3]['values'], dtype=torch.float32).view(-1, 1) # 250000
#        print(sample['state'][2]['values'].shape, sample['episode'])
        feature_graph.append(collecter)
        # collecterM.append(np.transpose(sample['state'][3]['incidence'])) 
        
        epi.append(sample['episode'])
        
        instances.append(sample['instance'])

        bobj.append(sample['b_obj'])
        
        i += 1

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)
    
    return feature_graph, np.stack(epi), np.stack(bobj), instances, np.stack(ini_sol)