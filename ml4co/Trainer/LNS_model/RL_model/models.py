import torch
import torch.nn as nn
from ml4co.Trainer.LNS_model.RL_model import model_utils
from ml4co.Trainer.LNS_trainer.RL.DataLoader import GNNDataLoader, GNNDataLoaderA

class Model(nn.Module):
    def __init__(self, name, network='gnn_actor', **network_kwargs):
        super().__init__()
        self.name = name
        self.network = model_utils.get_network_builder(network)(**network_kwargs)

class Actor(Model):
    def __init__(self, nb_actions, name='actor', network='gnn_actor', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions
        self.fc_out = nn.Linear(128, 1)

    def forward(self, obs):
        obs = obs.view(-1, 20)
        x = self.network(obs)
        x = torch.tanh(self.fc_out(x))
        return x.view(-1, 1000, 1)

class Actor_mean(Model):
    def __init__(self, device, nb_actions, name='actor', network='gnn_actor', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)

        self.nb_actions = nb_actions
        self.device = device
        self.fc1 = nn.Linear(64, 128)
        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)

        self.fc4 = nn.Linear(64, 64)
        self.ln4 = nn.LayerNorm(64)

        self.fc5 = nn.Linear(64, 32)
        self.ln5 = nn.LayerNorm(32)

        self.fc6 = nn.Linear(32, 1)

    def forward(self, obs):

        graph_num = len(obs)
        
        dataloader = GNNDataLoader(obs, graph_num, self.device)

        for batch in dataloader:

            x  = self.network(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                graph_num
            )

        # res1 = x

        # x = self.fc1(x)
        # x = torch.tanh(self.ln1(x))

        # x = self.fc2(x)
        # x = torch.tanh(self.ln2(x))

        # x = self.fc3(x)
        # x = torch.tanh(self.ln3(x))

        # x = x + res1  # 🔁 残差连接回原始输入维度

        # res2 = x

        # x = self.fc4(x)
        # x = torch.tanh(self.ln4(x))

        # x = x + res2  # 🔁 第二个残差连接（中间层）

        # x = self.fc5(x)
        # x = torch.tanh(self.ln5(x))

        # x = self.fc6(x)
        # x = torch.sigmoid(x)     # [0,1] 区间
        # x = x * 0.6 + 0.2         # 映射到 [0.2, 0.8]
        # x = torch.tanh(xc)
        # x = torch.sigmoid(xc)

        # x = (x+1)/2
        # x = torch.sigmoid(xc+1)      # 映射到[0,1]
        # x = (x - 0.5) * 1.6 + 0.5  # 拉伸范围到更宽，中心点是0.5
        # x = torch.clamp(x, 0, 1)  # 限制输出在[0,1]之间
                
        return x.view(-1, 1000, 1) # 1000*1

class Critic_mean(Model):
    def __init__(self, device, name='critic', network='gnn_critic', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.name = name
        self.device = device
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)

        # 归一化层
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(256)
    
    def forward(self, obs, action):

        graph_num = len(obs)
        
        dataloader = GNNDataLoaderA(obs, action, graph_num, self.device)

        for batch in dataloader:

            x = self.network(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                graph_num
            )

        # xc = self.ln1(x)
        # xc_c = torch.tanh(xc)
        # xc = self.fc1(xc_c)
        # xc = self.ln2(xc)
        # xc = torch.tanh(xc)

        # # xc_v = xc + x
        # xc = self.fc2(xc)
        # xc = self.ln2(xc)
        # xc = torch.mean(xc, dim=1)
        # xc = torch.tanh(xc)
        # xc = self.fc3(xc)
        # xc = self.ln3(xc)  # 重用已有的归一化层
        # x = torch.tanh(xc)
        # x = self.fc4(x)
        # x = torch.tanh(x)
        # x = self.fc5(x)

        return x

class Critic(nn.Module):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__()
        self.name = name
        self.layer_norm = True
        
        self.network_builder = model_utils.get_network_builder(network)(**network_kwargs)
        
        # 定义线性层
        self.output_layer1 = nn.Linear(64, 64)
        self.output_layer2 = nn.Linear(64, 1)
        
        # 初始化参数
        nn.init.uniform_(self.output_layer1.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer2.weight, -3e-3, 3e-3)
    
    def forward(self, obs, action):
            
        x = torch.cat([obs, action], dim=-1)  # 假设 obs 和 action 可以拼接
        x = x.view(-1, 21)
        x = self.network_builder(x)
        x = self.output_layer1(x)
        x = x.view(-1, 1000, 64)
        x = torch.mean(x, dim=1)
        x = self.output_layer2(x)
        return x
    
    @property
    def output_vars(self):
        return [param for name, param in self.named_parameters() if 'output' in name]
    
