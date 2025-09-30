import os
import datetime
import numpy as np
import random
import copy
import torch
import glob
import time
from rl4mip.DataCollector.LNS_data.RL_data.DataMemory import Memory
from rl4mip.DataCollector.LNS_data.RL_data.SFdata_Collector import collect_samples, collect_samples0
from rl4mip.Trainer.LNS_trainer.RL.TrainEnv import DDPG
from rl4mip.Trainer.LNS_model.RL_model.models import Actor_mean, Critic_mean
from rl4mip.Trainer.LNS_model.RL_model.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

class GNNPolicyTrainer:
    def __init__(self, device, method, problem, network_actor,
          network_critic, noise_type,
          normalize_returns,
          normalize_observations, load_path,
          model_dir,
          param_noise_adaption_interval,
          exploration_strategy):
        
        self.device = device
        self.method = method
        self.problem = problem
        self.network_actor = network_actor
        self.network_critic = network_critic
        self.noise_type = noise_type
        self.normalize_returns = normalize_returns
        self.normalize_observations = normalize_observations
        self.load_path = load_path
        self.model_dir = model_dir
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.exploration_strategy = exploration_strategy
        self.model_path = os.path.join(model_dir, method, problem, network_actor)
        os.makedirs(self.model_path, exist_ok=True)

    def train(self, data_dir, nb_epochs, nb_rollout_steps,
          observation_range, 
          action_range,
          return_range,
          reward_scale, critic_l2_reg,
          actor_lr,
          critic_lr,
          popart,
          gamma,  
          clip_norm,
          nb_train_steps,
          nb_eval_steps, #20 #50  
          batch_size, # per MPI worker  64 32  64   128   64  128 300
          tau, 
          batch_sample,
          time_limit,
          eval_val,
          seed):
        
        log(f"max_epochs: {nb_epochs}")
        log(f"batch_size: {batch_size}")
        log(f"actor_lr: {actor_lr}")
        log(f"critic_lr: {critic_lr}")
        log(f"seed {seed}")
        log(f"device: {self.device}")

        problem = self.problem
        method = self.method
        noise_type = self.noise_type
        normalize_returns = self.normalize_returns
        normalize_observations = self.normalize_observations
        device = self.device

        if problem == 'setcover':
            nvars = 1000
            ncons = 5000
            dens=0.05

        elif problem == 'indset':
            nvars = 1500
            ncons = 5939
            # affinity=4
            dens = 4
        elif problem == 'cauctions':
            nvars = 4000
            ncons = 2674
            # add_item_prob=0.7
            dens = 0.7
        elif problem == 'maxcut':
            nvars = 2975
            ncons = 4950
            # ratio = 5
            dens = 5
        else:
            print("there does not exist such problem to address")

        instance_train_dir = os.path.join(data_dir, 'instances', f'{method}/{problem}/train')
        instance_valid_dir = os.path.join(data_dir, 'instances', f'{method}/{problem}/valid')

        instances_train = glob.glob(f'{instance_train_dir}/*.lp')
        instances_valid = glob.glob(f'{instance_valid_dir}/*.lp')
        nb_actions = nvars

        nb_epoch_cycles = len(instances_train)//batch_sample 

        nenvs = batch_sample

        rng = np.random.RandomState(seed)

        out_dir = f'./samples/{problem}/'

        print('nb_epochs', nb_epochs, 'nb_epoch_cycles', nb_epoch_cycles, 'nb_rollout_steps', nb_rollout_steps)
        print("{} train instances for {} samples".format(len(instances_train),nb_epoch_cycles*nb_epochs*batch_sample))

        network_actor = self.network_actor
        network_critic = self.network_critic

        actor = Actor_mean(device, nb_actions, network=network_actor).to(device)
        critic = Critic_mean(device, network=network_critic).to(device)

        memory = Memory(limit=int(750), action_shape=(nb_actions,1,), observation_shapes={
            'variable_features': (nvars, 22),
            'constraint_features': (ncons, 14),
            'edge_indices': (2, int(nvars*ncons*dens)),
            'edge_features': (int(nvars*ncons*dens), 1)
        })

        action_noise = None
        param_noise = None

        if self.noise_type is not None:
            for current_noise_type in noise_type.split(','):
                current_noise_type = current_noise_type.strip()
                if current_noise_type == 'none':
                    pass
                elif 'adaptive-param' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
                elif 'normal' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                elif 'ou' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

        agent = DDPG(device, actor, critic, memory, [nvars, ncons, int(nvars*ncons*dens)], action_noise=action_noise, 
                     param_noise=param_noise, action_shape = (nb_actions,), gamma=gamma, tau=tau, 
                     normalize_returns=normalize_returns, enable_popart = popart, normalize_observations=normalize_observations,
                     batch_size=batch_size, observation_range=observation_range, action_range=action_range, return_range=return_range,
                     critic_l2_reg=critic_l2_reg, actor_lr=actor_lr, 
                     critic_lr=critic_lr, clip_norm=clip_norm, reward_scale=reward_scale)

        agent = agent.to(device)

        if self.load_path is not None:
            agent.load(self.load_path)

        agent.reset()
        
        min_obj = 1000000

        for epoch in range(nb_epochs):

            print(f"------------------------------start {epoch}th train------------------------")
            
            random.shuffle(instances_train)
            
            for cycle in range(nb_epoch_cycles):
                
                print(f"------------------------------start {epoch}th train {cycle}th data------------------------")

                formu_feat, epi, ori_objs, instances, ini_sol = collect_samples0(instances_train, out_dir + '/train', rng, batch_sample,
                            exploration_policy=self.exploration_strategy,
                            batch_id=cycle,
                            eval_flag=eval_val,
                            time_limit=None)
                if nenvs > 1:
                    agent.reset()

                cur_sols = ini_sol
                
                pre_sols = np.zeros([2, batch_sample, nb_actions])

                rec_inc = [[] for _ in range(batch_sample)] 
                [rec_inc[r].append(ini_sol[r]) for r in range(batch_sample)]  #ADD    
                rec_best = np.copy(ori_objs)                 #ADD
                inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample)]) 
                avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample)]) 

                formu_feat_ch = copy.deepcopy(formu_feat)

                for i in range(len(formu_feat)):
                    formu_feat_ch[i]['variable_features'] = torch.tensor(np.concatenate((formu_feat[i]['variable_features'], inc_val[i][:,np.newaxis], avg_inc_val[i][:,np.newaxis], pre_sols.transpose(1,2,0)[i], cur_sols[i][:,np.newaxis]), axis=-1), dtype=torch.float32)   # 修改特征表达
                
                cur_obs = formu_feat_ch

                for t_rollout in range(nb_rollout_steps):

                    print(f"-----------------------------start {epoch}th train {cycle}th data {t_rollout} rollout-----------------------")

                    action, _, _, _ = agent.step(cur_obs, apply_noise=True, compute_Q=True) 
                    
                    pre_sols = np.concatenate((pre_sols, cur_sols[np.newaxis,:,:]), axis=0) 

                    # actionn = action
                    # actionn=np.where(actionn > 0.5, actionn, 0.)  
                    # actionn=np.where(actionn == 0., actionn, 1.) 

                    next_sols, epi, cur_objs, instances, _ = collect_samples(instances, epi, cur_sols, action, out_dir + '/train', rng, batch_sample,
                                    exploration_policy=self.exploration_strategy,
                                    eval_flag=eval_val,
                                    time_limit=time_limit) 
        
                    cur_sols=next_sols.copy()

                    if t_rollout>0:
                        agent.store_transition(cur_obs_s, action_s, r_s, next_obs_s, action, epi)
                    
                    r = ori_objs - cur_objs

                    print("r", r)        
                                                                
                    inc_ind = np.where(cur_objs < rec_best)[0]       
                    [rec_inc[r].append(cur_sols[r]) for r in inc_ind]                       
                    rec_best[inc_ind] = cur_objs[inc_ind]                  
                                        

                    inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample)])
                    avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample)])            
                    formu_feat_ch = copy.deepcopy(formu_feat)

                    for i in range(len(formu_feat)):
                        formu_feat_ch[i]['variable_features'] = torch.tensor(np.concatenate((formu_feat[i]['variable_features'], inc_val[i][:,np.newaxis], avg_inc_val[i][:,np.newaxis], pre_sols[1:3].transpose(1,2,0)[i], cur_sols[i][:,np.newaxis]), axis=-1), dtype=torch.float32)   # 修改特征表达
                    
                    next_obs = formu_feat_ch
                    del formu_feat_ch
                    torch.cuda.empty_cache()
                    cur_obs_s = cur_obs.copy()
                    action_s = action.copy()
                    r_s = r
                    next_obs_s = next_obs.copy()
                                
                    cur_obs = next_obs               
                    ori_objs = cur_objs

                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                
                for t_train in range(nb_train_steps):
                    
                    if memory.nb_entries < batch_size:
                        print(f"[Train] dump:memory size = {memory.nb_entries} < batch size = {batch_size}")
                        eval = 0
                        break
                    else:
                        eval = 1 

                    if t_train % self.param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)
                    
                    cl, al = agent.Train(device)

                    agent.update_target_net()
                    
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)

                if eval == 1:

                    print(f"--------------------{epoch}th {cycle}th data update network--------------------")

                    obj_lis = []

                    for cyc in range(len(instances_valid)):

                        print(f"--------------------开始验证{epoch}th {cycle}th 后 第{cyc}个例子结果--------------------")
                        
                        batch_sample_eval = 1

                        eval_st_time = time.time()

                        formu_feat, epi, ori_objs, instances, ini_sol= collect_samples0([instances_valid[cyc]], out_dir + '/valid', rng, batch_sample_eval,
                                        exploration_policy=self.exploration_strategy,
                                        batch_id=0,
                                        eval_flag=eval_val,
                                        time_limit=None) 

                        cur_sols = ini_sol
                        pre_sols = np.zeros([2, batch_sample_eval, 1000]) 

                        rec_inc = [[] for r in range(batch_sample_eval)] 
                        
                        [rec_inc[r].append(ini_sol[r]) for r in range(batch_sample_eval)] 
                        rec_best = np.copy(ori_objs)              
                        inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample_eval)]) 
                        avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample_eval)]) 
                    
                        formu_feat_ch = copy.deepcopy(formu_feat)

                        for i in range(len(formu_feat)):
                            formu_feat_ch[i]['variable_features'] = torch.tensor(np.concatenate((formu_feat[i]['variable_features'], inc_val[i][:,np.newaxis], avg_inc_val[i][:,np.newaxis], pre_sols.transpose(1,2,0)[i], cur_sols[i][:,np.newaxis]), axis=-1), dtype=torch.float32)   # 修改特征表达
                        
                        cur_obs = formu_feat_ch  

                        for t_rollout in range(nb_eval_steps):       
                        
                            action, q, _, _ = agent.step(cur_obs, apply_noise=True, compute_Q=True) 
                            
                            pre_sols = np.concatenate((pre_sols, cur_sols[np.newaxis,:,:]), axis=0) 

                            actionn=np.copy(action)
                            actionn=np.where(actionn > 0.5, actionn, 0.)  
                            actionn=np.where(actionn == 0., actionn, 1.) 

                            next_sols, epi, cur_objs, instances, _ = collect_samples(instances, epi, cur_sols, action, out_dir + '/train', rng, batch_sample_eval,
                                            exploration_policy=self.exploration_strategy,
                                            eval_flag=eval_val,
                                            time_limit=time_limit)
                             
                            cur_sols=next_sols.copy()

                            inc_ind = np.where(cur_objs < rec_best)[0]    
                            
                            print(f"{cyc} 当前 cur_objs", cur_objs)

                            [rec_inc[r].append(cur_sols[r]) for r in inc_ind]       

                            inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample_eval)]) 
                            avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample_eval)])                 

                            formu_feat_ch = copy.deepcopy(formu_feat)

                            for i in range(len(formu_feat)):
                                formu_feat_ch[i]['variable_features'] = torch.tensor(np.concatenate((formu_feat[i]['variable_features'], inc_val[i][:,np.newaxis], avg_inc_val[i][:,np.newaxis], pre_sols[1:3].transpose(1,2,0)[i], cur_sols[i][:,np.newaxis]), axis=-1), dtype=torch.float32)   # 修改特征表达
                            
                            next_obs = formu_feat_ch
                                    
                            cur_obs = next_obs               
                            step_time = time.time()

                            print('----------------step:', {t_rollout}, 'time:', f'{step_time - eval_st_time}', f'{t_rollout}', 'obj', cur_objs,'--------------')
                            
                            rec_best[inc_ind] = cur_objs[inc_ind]  
                            print("rec_best:", rec_best)
                            if step_time - eval_st_time > 200:
                                break
                            else:
                                continue

                        print('time_______________________________________')     
                        print(time.time()-eval_st_time)       
                        obj_lis.append(rec_best[0])

                    ave = np.mean(obj_lis)
                    
                    print(f"----------------------------------------------average obj:{ave}---------------------------------------")

                    if self.model_path is not None and ave < min_obj:
                        s_path = os.path.join(self.model_path, f'{problem}.pt')

                        # os.makedirs(os.path.dirname(s_path), exist_ok=True)

                        print(f"Saving improved model to: {s_path} (update current value: {ave:.4f} < best so far: {min_obj:.4f})")
                        
                        agent.save(s_path)
                        
                        min_obj = ave
                    else:
                        print(f"Not saving improved model to: {s_path} (updatae current value: {ave:.4f} > best so far: {min_obj:.4f})")

                    break