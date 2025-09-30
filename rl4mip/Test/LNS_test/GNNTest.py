import os
import datetime
import numpy as np
import random
import copy
import torch
import time
from rl4mip.DataCollector.LNS_data.CL_data.utils import logger
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

class GNNPolicyTester:
    def __init__(self, device, data_dir, method, problem, network_actor,
          network_critic, noise_type,
          normalize_returns,
          normalize_observations, load_path,
          model_dir,
          param_noise_adaption_interval,
          exploration_strategy):
        
        self.data_dir = data_dir
        self.device = device
        self.problem = problem
        self.method = method
        self.network_actor = network_actor
        self.network_critic = network_critic
        self.noise_type = noise_type
        self.normalize_returns = normalize_returns
        self.normalize_observations = normalize_observations
        self.load_path = load_path
        self.model_dir = model_dir
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.exploration_strategy = exploration_strategy
        # self.model_path = os.path.join(model_dir, problem, network_actor)
        # os.makedirs(self.model_path, exist_ok=True)

    def test(self, instance, nb_epochs,
          observation_range, 
          action_range,
          return_range,
          reward_scale, critic_l2_reg,
          actor_lr,
          critic_lr,
          popart,
          gamma,  
          clip_norm,
          nb_eval_steps, #20 #50  
          batch_size, # per MPI worker  64 32  64   128   64  128 300
          tau, 
          time_limit,
          eval_val,
          seed,
          results_loc,
          instance_id):
        

        log(f"max_epochs: {nb_epochs}")
        log(f"batch_size: {batch_size}")
        log(f"actor_lr: {actor_lr}")
        log(f"critic_lr: {critic_lr}")
        log(f"seed {seed}")
        log(f"device: {self.device}")

        problem = self.problem
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
            dens=4

        elif problem == 'cauctions':
            nvars = 4000
            ncons = 2674
            add_item_prob=0.7
            dens=0.7

        elif problem == 'maxcut':
            nvars = 2975
            ncons = 4950
            ratio = 5
            dens=5
        else:
            print("there does not exist such problem to address")
            
        nb_actions = nvars

        # nb_epoch_cycles = len(instances_train)//batch_sample 

        rng = np.random.RandomState(seed)

        out_dir = f'./samples/{problem}/'

        # print('nb_epochs', nb_epochs, 'nb_epoch_cycles', nb_epoch_cycles, 'nb_rollout_steps', nb_rollout_steps)
        # print("{} train instances for {} samples".format(len(instances_train),nb_epoch_cycles*nb_epochs*batch_sample))

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

        if self.load_path is None:
            # print(self.load_path, problem, network_actor, f'{problem}.pt')
            # model_path = os.path.join(self.load_path, problem, network_actor, f'{problem}.pt')
            agent.load(self.model_dir)

        agent.reset()

        obj_lis = []

        # instances_test = os.path.join(self.data_dir, 'instances', self.method, problem, 'test')


        batch_sample_eval = 1

        eval_st_time = time.time()

        formu_feat, epi, ori_objs, instances, ini_sol= collect_samples0([instance], out_dir + '/test', rng, batch_sample_eval,
                        exploration_policy=self.exploration_strategy,
                        batch_id=0,
                        eval_flag=eval_val,
                        time_limit=None) 

        cur_sols = ini_sol
        pre_sols = np.zeros([2,batch_sample_eval,1000]) 

        rec_inc = [[] for r in range(batch_sample_eval)] 
        
        [rec_inc[r].append(ini_sol[r]) for r in range(batch_sample_eval)] 
        rec_best = np.copy(ori_objs)              
        inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample_eval)]) 
        avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample_eval)]) 
    
        formu_feat_ch = copy.deepcopy(formu_feat)
        step0_time = time.time()
        log_entry = dict()

        times = []
        obj_lis.append(rec_best[0])
        times.append(step0_time - eval_st_time)

        log_entry['iteration_time'] = step0_time - eval_st_time
        log_entry['run_time'] = step0_time - eval_st_time
        log_entry['primal_bound'] = ori_objs
        log_entry['best_primal_scip_sol'] = cur_sols

        LNS_log = [log_entry]

        for i in range(len(formu_feat)):
            formu_feat_ch[i]['variable_features'] = torch.tensor(np.concatenate((formu_feat[i]['variable_features'], inc_val[i][:,np.newaxis], avg_inc_val[i][:,np.newaxis], pre_sols.transpose(1,2,0)[i], cur_sols[i][:,np.newaxis]), axis=-1), dtype=torch.float32)   # 修改特征表达
        
        cur_obs = formu_feat_ch  


        # print("start_time_points: ", log_entry['run_time'])
        # print("start_primal_bounds: ", LNS_log[0]['primal_bound'])
        logger(f"Problem {instance_id}: start_time_points: {log_entry['run_time']}", logfile=results_loc)
        logger(f"Problem {instance_id}: start_primal_bounds: {LNS_log[0]['primal_bound']}", logfile=results_loc)

        for t_rollout in range(nb_eval_steps):       
            
            logger(f"Problem {instance_id}: start solve step: {t_rollout}", logfile = results_loc)

            start_time = time.time()
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
            # print("cur_objs", cur_objs)
            [rec_inc[r].append(cur_sols[r]) for r in inc_ind]       

            inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample_eval)]) 
            avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample_eval)])                 

            formu_feat_ch = copy.deepcopy(formu_feat)

            for i in range(len(formu_feat)):
                formu_feat_ch[i]['variable_features'] = torch.tensor(np.concatenate((formu_feat[i]['variable_features'], inc_val[i][:,np.newaxis], avg_inc_val[i][:,np.newaxis], pre_sols[1:3].transpose(1,2,0)[i], cur_sols[i][:,np.newaxis]), axis=-1), dtype=torch.float32)   # 修改特征表达
            
            next_obs = formu_feat_ch
                    
            cur_obs = next_obs

            step_time = time.time()

            # log_entry = dict()
            log_entry['iteration_time'] = step_time - start_time
            log_entry['run_time'] = step_time - eval_st_time
            log_entry['primal_bound'] = cur_objs
            log_entry['best_primal_scip_sol'] = cur_sols
            
            # print('----------------step:', {t_rollout}, 'time:', f'{step_time - eval_st_time}', f'{t_rollout}', 'obj', cur_objs,'--------------')
            logger(f"Problem {instance_id}: Finished LNS step {t_rollout}: obj_val = {log_entry['primal_bound']} in iteration time {log_entry['iteration_time']}", logfile = results_loc)
            times.append(step_time - eval_st_time)
            rec_best[inc_ind] = cur_objs[inc_ind]  

            LNS_log.append(log_entry)


            # print("rec_best:", rec_best)
            if step_time - eval_st_time > 200:
                break
            else:
                continue


            

        # print('time_______________________________________')     
        # print(time.time()-eval_st_time)       
        obj_lis.append(rec_best[0])
        # print("obj_lis:", obj_lis)
        if times and obj_lis:
            logger(
                f"Problem {instance_id}: initial solution obj = {obj_lis[0]}, "
                f"found in time {times[0]:.2f}s",
                logfile=results_loc
            )

        # 记录最优解信息（从所有时间点中找最优）
        
        if obj_lis:
            if problem == "cauctions" or problem == "indset":
                best_idx = np.argmax(obj_lis)  # 假设是最小化问题
            elif problem == "setcover" or problem == "maxcut":
                best_idx = np.argmin(obj_lis)  # 假设是最大化问题
            else:
                raise ValueError(f"Unknown problem type: {problem}")
            logger(
                f"Problem {instance_id}: best primal bound {obj_lis[best_idx]}, "
                f"found at time {times[best_idx]:.2f}s, "
                f"total time {times[-1]:.2f}s",
                logfile=results_loc
            )  

        return LNS_log, times, obj_lis
