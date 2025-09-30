import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ml4co.Trainer.LNS_trainer.RL import env_utils
import copy
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)

def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def reduce_var(x, axis=None, keepdims=False):
    mean = torch.mean(x, dim=axis, keepdim=keepdims)
    devs_squared = (x - mean) ** 2
    return torch.mean(devs_squared, dim=axis, keepdim=keepdims)

class DDPG(object):
    def __init__(self, device, actor, critic, memory, observation_shape, action_shape, param_noise, action_noise,
        gamma, tau, normalize_returns, enable_popart, normalize_observations,
        batch_size, observation_range, action_range, return_range,
        critic_l2_reg, actor_lr, critic_lr, clip_norm, reward_scale):

        super(DDPG, self).__init__()

        self.obs0 = [{
            'variable_features': torch.zeros((observation_shape[0], 22), dtype=torch.float32),
            'constraint_features': torch.zeros((observation_shape[1], 14), dtype=torch.float32),
            'edge_indices': torch.zeros((2, observation_shape[2]), dtype=torch.long),
            'edge_features': torch.zeros((observation_shape[2], 1), dtype=torch.float32)
        }]

        self.obs1 = [{
            'variable_features': torch.zeros((observation_shape[0], 22), dtype=torch.float32),
            'constraint_features': torch.zeros((observation_shape[1], 14), dtype=torch.float32),
            'edge_indices': torch.zeros((2, observation_shape[2]), dtype=torch.long),
            'edge_features': torch.zeros((observation_shape[2], 1), dtype=torch.float32)
        }]

        self.terminals1 = torch.zeros((1, 1), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((1, 1), dtype=torch.float32, device=device)
        self.actions = torch.zeros((1,) + action_shape + (1,), dtype=torch.float32, device=device)
        self.critic_target = torch.zeros((1, 1), dtype=torch.float32, device=device)
        self.param_noise_stddev = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.next_actions = torch.zeros((1,) + action_shape + (1,), dtype=torch.float32, device=device)
        self.Q0 = torch.zeros((1, 1), dtype=torch.float32, device=device)        
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        normalized_obs0 = self.obs0
        normalized_obs1 = self.obs1

        if self.normalize_returns:
            self.ret_rms = env_utils.RunningMeanStd()
        else:
            self.ret_rms = None

        target_actor = copy.deepcopy(actor)
        
        target_actor.load_state_dict(actor.state_dict())

        target_actor.name = 'target_actor'

        self.target_actor = target_actor

        target_critic = copy.deepcopy(critic)

        target_critic.load_state_dict(critic.state_dict())
        
        target_critic.name = 'target_critic'

        self.target_critic = target_critic
        
        self.actor_tf = actor(normalized_obs0)

        self.actor_tf_next = actor(normalized_obs1)

        self.normalized_critic_with_actor_tf = self.critic(normalized_obs0, self.actor_tf)

        self.critic_with_actor_tf = denormalize(
            torch.clamp(self.normalized_critic_with_actor_tf,
                    self.return_range[0],
                    self.return_range[1]),
            self.ret_rms
        )

        self.target_act = target_actor(normalized_obs1)
        Q_obs1 = denormalize(target_critic(normalized_obs1, self.next_actions), self.ret_rms).to(device)
        self.target_Q = self.rewards + gamma * Q_obs1
        self.Q_obs0 = denormalize(critic(normalized_obs0, self.actions), self.ret_rms).to(device)      #QA2C

        self.choice = torch.cat([1 - self.actor_tf, self.actor_tf], dim=2)
        self.choice1 = self.choice.view(-1, 2)
        self.indice = torch.cat([
            torch.arange(1000, dtype=torch.int64).unsqueeze(-1).to(device), 
            self.actions.reshape(-1, 1).to(torch.int64).to(device)
        ], dim=-1)

        self.decision = self.choice1[self.indice[:, 0], self.indice[:, 1]]

        self.decision1 = self.decision.view(-1, 1000, 1)

        # 损失函数（注意负号方向）
        self.actor_loss = -torch.mean(torch.sum(torch.log(self.decision1 + 1e-10), dim=1) * self.Q0)

        # 参数统计
        actor_shapes = [tuple(param.shape) for param in self.actor.parameters()]
        actor_nb_params = sum(np.prod(shape) for shape in actor_shapes)
        logger.info('actor shapes: {}'.format(actor_shapes))
        logger.info('actor params: {}'.format(actor_nb_params))

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def compute_actor_loss(self, batch, Q0):

        result_obs0 = []
        
        for i in range(batch['obs0']['variable_features'].shape[0]):
            item = {key: torch.tensor(value[i]) for key, value in batch['obs0'].items()}
            result_obs0.append(item)

        # obs0 = torch.cat((batch['obs0']), dim=-1)
        inter = self.actor(result_obs0)  # Actor 计算 inter
        inter_mean = inter.mean(dim=1)
        actor_loss = -torch.mean(inter_mean * Q0)  # 计算 actor_loss
        return actor_loss

    def step(self, obs, apply_noise=True, compute_Q=True):

        if self.param_noise is not None and apply_noise:
            actor_model = self.perturbed_actor
        else:
            actor_model = self.actor
            
        with torch.no_grad():
            action = actor_model(obs)
        
        if compute_Q:
            q_value = self.critic(obs, action)
        else:
            q_value = None
        
        if self.action_noise is not None and apply_noise:
            noise = torch.as_tensor(self.action_noise(), dtype=torch.float32)
            noise = noise.unsqueeze(1)  
            action += noise
        
        action = torch.clamp(action, self.action_range[0], self.action_range[1])

        return action.cpu().numpy(), q_value.cpu().detach().numpy() if q_value is not None else None, None, None
    
    def next_step(self, obs, apply_noise=True, compute_Q=False):

        target_actor = self.target_actor
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = target_actor(obs_tensor)

        if compute_Q:
            q_value = self.target_critic(obs_tensor, action)
        else:
            q_value = None

        if self.action_noise is not None and apply_noise:
            noise = torch.as_tensor(self.action_noise(), dtype=torch.float32)
            action += noise

        action = torch.clamp(action, self.action_range[0], self.action_range[1])
        return action.cpu().numpy(), q_value.cpu().numpy() if q_value else None, None, None
    
    def store_transition(self, obs0, action, reward, obs1, action_next, ins):

        reward *= self.reward_scale

        for b in range(len(obs0)):
            self.memory.append(
                obs0[b],
                action[b],
                reward[b],
                obs1[b],
                action_next[b],
                ins[b]
            )

            if self.normalize_observations:
                self.obs_rms.update(obs0[b].cpu().numpy())

    def save(self, save_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.target_actor.state_dict(),
            'critic_target': self.target_critic.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['actor_target'])
        self.target_critic.load_state_dict(checkpoint['critic_target'])

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.critic_target.to(device)

        return self
    
    def Train(self, device):
        
        # batch = self.memory.sample(batch_size=self.batch_size)
        
        # result_obs0 = []
        
        # for i in range(batch['obs0']['variable_features'].shape[0]):
        #     item = {key: torch.tensor(value[i]) for key, value in batch['obs0'].items()}
        #     result_obs0.append(item)

        # result_obs1 = []
        
        # for i in range(batch['obs1']['variable_features'].shape[0]):
        #     item = {key: torch.tensor(value[i]) for key, value in batch['obs1'].items()}
        #     result_obs1.append(item)

        # if self.normalize_returns and self.enable_popart:

        #     Q_obs1 = self.target_critic(result_obs0, batch['next_actions'])

        #     old_mean, old_std = self.ret_rms.mean, self.ret_rms.std

        #     target_Q = batch['rewards'] + self.gamma * Q_obs1
        #     self.ret_rms.update(target_Q.flatten().detach().cpu().numpy())  # 假设 ret_rms 需要 NumPy 数组

        #     # 重新标准化 Q 值
        #     self.ret_rms.mean = old_mean
        #     self.ret_rms.std = old_std

        # else:

        #     batch['next_actions'] = torch.tensor(batch['next_actions']).float() 

        #     target_Q = self.target_critic(result_obs1, batch['next_actions'])
        #     target_Q = torch.tensor(batch['rewards']).to(device) + self.gamma * target_Q

        #     Q0 = self.critic(result_obs0, torch.tensor(batch['actions'], dtype=torch.float32)).detach()

        # print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')

        # print(batch['rewards'].shape)

        # print(Q0.shape)

        # print(target_Q.shape)

        # print(batch['actions'].shape)

        # critic_loss = nn.MSELoss()(target_Q, Q0)

        # if self.critic_l2_reg > 0.:
        #     critic_reg = 0.
        #     critic_reg_vars = [param for name, param in self.critic.named_parameters() if "weight" in name and "output" not in name]
            
        #     for var in critic_reg_vars:
        #         logger.info(f'  Regularizing: {var.shape}')
        #         critic_reg += torch.sum(var ** 2) 
            
        #     logger.info(f'  Applying L2 regularization with {self.critic_l2_reg}')
        #     critic_loss += self.critic_l2_reg * critic_reg 

        # critic_nb_params = sum(p.numel() for p in self.critic.parameters())
        # logger.info(f'  Critic params: {critic_nb_params}')

        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)

        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # actor_loss = self.compute_actor_loss(batch,
        #     self.critic(result_obs0,
        #                 torch.tensor(batch['actions'], dtype=torch.float32).detach())
        # )

        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

            # 取出 batch
        batch = self.memory.sample(batch_size=self.batch_size)

        # 转换 obs0 和 obs1 为 tensor 列表，避免循环构建
        result_obs0 = [
            {key: torch.tensor(val[i], dtype=torch.float32, device=device) for key, val in batch['obs0'].items()}
            for i in range(batch['obs0']['variable_features'].shape[0])
        ]

        result_obs1 = [
            {key: torch.tensor(val[i], dtype=torch.float32, device=device) for key, val in batch['obs1'].items()}
            for i in range(batch['obs1']['variable_features'].shape[0])
        ]

        next_actions = torch.tensor(batch['next_actions'], dtype=torch.float32, device=device)
        actions = torch.tensor(batch['actions'], dtype=torch.float32, device=device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=device)

        if self.normalize_returns and self.enable_popart:
            Q_obs1 = self.target_critic(result_obs0, next_actions)

            # popart 标准化
            old_mean, old_std = self.ret_rms.mean, self.ret_rms.std
            target_Q = rewards + self.gamma * Q_obs1
            self.ret_rms.update(target_Q.detach().cpu().numpy())
            self.ret_rms.mean, self.ret_rms.std = old_mean, old_std

        else:
            target_Q = rewards + self.gamma * self.target_critic(result_obs1, next_actions)

        Q0 = self.critic(result_obs0, actions).detach()

        # 计算 critic loss
        critic_loss = nn.MSELoss()(target_Q, Q0)

        if self.critic_l2_reg > 0.:
            critic_reg = sum((param ** 2).sum() for name, param in self.critic.named_parameters()
                            if "weight" in name and "output" not in name)
            critic_loss += self.critic_l2_reg * critic_reg

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        # 清空优化器
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor loss
        with torch.no_grad():
            actions_detached = actions.detach()

        Q_for_actor = self.critic(result_obs0, actions_detached)
        actor_loss = self.compute_actor_loss(batch, Q_for_actor)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 清理中间变量，节省显存
        del result_obs0, result_obs1, target_Q, Q0, Q_for_actor
        torch.cuda.empty_cache()

        return critic_loss.item(), actor_loss.item()

    def initialize(self):

        self.soft_update(self.actor, self.target_actor, tau=1.0)
        self.soft_update(self.critic, self.target_critic, tau=1.0)

    def update_target_net(self):

        self.soft_update(self.actor, self.target_actor, tau=self.tau)
        self.soft_update(self.critic, self.target_critic, tau=self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):

            target_param.data.copy_(tau * local_param.data.detach() + (1.0 - tau) * target_param.data)

    def get_stats(self):

        if self.stats_sample is None:
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)

        obs0 = torch.FloatTensor(self.stats_sample['obs0'])
        actions = torch.FloatTensor(self.stats_sample['actions'])
        
        stats = {}
        with torch.no_grad():

            if self.normalize_returns:
                stats['ret_rms_mean'] = self.ret_rms.mean.cpu().numpy()
                stats['ret_rms_std'] = self.ret_rms.std.cpu().numpy()

            if self.normalize_observations:
                stats['obs_rms_mean'] = torch.mean(self.obs_rms.mean).cpu().numpy()
                stats['obs_rms_std'] = torch.mean(self.obs_rms.std).cpu().numpy()

            critic_out = self.critic(obs0, actions)
            stats['reference_Q_mean'] = torch.mean(critic_out).cpu().numpy()
            stats['reference_Q_std'] = torch.std(critic_out).cpu().numpy()

            actor_out = self.actor(obs0)
            stats['reference_action_mean'] = torch.mean(actor_out).cpu().numpy()
            stats['reference_action_std'] = torch.std(actor_out).cpu().numpy()

        if self.param_noise is not None:
            stats.update(self.param_noise.get_stats())
        
        return stats

    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.0
        
        # 采样批次数据
        batch = self.memory.sample(batch_size=self.batch_size)
        obs0 = torch.FloatTensor(batch['obs0'])
        # 扰动自适应策略网络（深拷贝）
        perturbed_actor = copy.deepcopy(self.actor)
        for param in perturbed_actor.parameters():
            param.data += torch.randn_like(param) * self.param_noise.current_stddev
        
        # 计算策略距离（L2范数）
        with torch.no_grad():
            orig_actions = self.actor(obs0)
            perturbed_actions = perturbed_actor(obs0)
            distance = torch.mean(torch.norm(orig_actions - perturbed_actions, dim=1))
        
        # 分布式平均（可选）
        if self.use_mpi:
            import torch.distributed as dist
            dist.all_reduce(distance, op=dist.ReduceOp.SUM)
            distance /= dist.get_world_size()
        
        # 调整噪声强度
        self.param_noise.adapt(distance.item())
        return distance.item()
    
    def reset(self):

        if self.action_noise is not None:
            self.action_noise.reset_state()

        if self.param_noise is not None:
            with torch.no_grad():
                for param in self.actor.parameters():
                    noise = torch.randn_like(param) * self.param_noise.current_stddev
                    param.data += noise