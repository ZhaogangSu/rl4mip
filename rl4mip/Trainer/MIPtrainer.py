import os
import sys
import numpy as np

from rl4mip.Trainer.Nodeselect_trainer.ML.ml_node_train import ML_Nodeselect_Trainer
from rl4mip.Trainer.Nodeselect_trainer.DSO.dso_node_train import TrainDSOAgent

from rl4mip.Trainer.Branch_trainer.GNN.GNNBranch import GNNBranchTrainer
from rl4mip.Trainer.Branch_trainer.Hybrid.HybridBranch import HybridBranchTrainer
from rl4mip.Trainer.Branch_trainer.Symb.SymbBranch import SymbBranchTrainer

from rl4mip.Trainer.LNS_trainer.CL.CL_LNS import CLTrainer
from rl4mip.Trainer.LNS_trainer.IL.IL_LNS import ILTrainer, ILTrainer_Pre
from rl4mip.Trainer.LNS_trainer.RL.GNNTrain import GNNPolicyTrainer
from rl4mip.Trainer.LNS_trainer.GBDT.TrainGNN import trainGNN
from rl4mip.Trainer.LNS_trainer.GBDT.TrainGBDT import trainGBDT

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class MIPTrain:
    def __init__(self, model_dir, data_dir, device):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = device

    def train(self, task, problem, method):
        if task == 'node_selection':
            if method == 'gnn' or method == 'ranknet' or method == 'svm':
                node_ml_trainer = ML_Nodeselect_Trainer(problem=problem, datapath=self.data_dir, model_dir=self.model_dir, device=self.device)
                return node_ml_trainer.train(method=method, lr=0.005, n_epoch=2, patience=10, early_stopping=20, normalize=True,  
                                batch_train=16, batch_valid=256, loss_fn=torch.nn.BCELoss(), optimizer_fn=torch.optim.Adam)
            elif method == 'symb':
                node_dso_trainer = TrainDSOAgent(total_iter=500,
                                            batch_size=1024,
                                            data_batch_size=1000,
                                            eval_expression_num=48,
                                            record_expression_num=16,
                                            record_expression_freq=10,
                                            early_stop=1000,
                                            continue_train_path=None,
                                            datapath=self.data_dir,
                                            model_dir=self.model_dir,
                                            device=self.device,
                                            instance_kwargs={'instance_type': problem, 'train_num': 100000, 'valid_num': 10000},
                                            expression_kwargs={'math_list': 'all', 'var_list': 'full'},
                                            dso_agent_kwargs={'min_length': 4, 'max_length': 10, 'hidden_size': 128, 'num_layers': 2},
                                            rl_algo_kwargs={'class': 'PPOAlgo', 'kwargs': {'lr_actor': 5e-05, 'K_epochs': 8, 'entropy_coef': 0.05, 'entropy_gamma': 0.9, 'entropy_decrease': False, 'lr_decrease': False, 'decrease_period': 700}}
                                            )
                return node_dso_trainer.train()
        
        elif task == 'branch':
            if method == 'gnn':
                branch_gnn_trainer = GNNBranchTrainer( problem=problem, datapath=self.data_dir, 
                                                model_dir=self.model_dir, device=self.device,
                                                batch_size=32, pretrain_batch_size=128, valid_batch_size=128)
                return branch_gnn_trainer.train()
            elif method == 'hybrid':
                branch_hybrid_trainer = HybridBranchTrainer(  problem=problem, datapath=self.data_dir, 
                                                    model_dir=self.model_dir, device=self.device,
                                                    epoch_size=312, batch_size=32, pretrain_batch_size=128, valid_batch_size=128,
                                                    no_e2e=False, distilled=True, AT='MHE', beta_at=0.0001)
                return branch_hybrid_trainer.train()
            elif method == 'symb':
                branch_symb_trainer = SymbBranchTrainer( problem=problem, datapath=self.data_dir, 
                                                model_dir=self.model_dir, device=self.device,
                                                early_stop=3)
                return branch_symb_trainer.train()
            elif method == 'graph':
                branch_graph_trainer = SymbBranchTrainer( problem=problem, datapath=self.data_dir, 
                                                model_dir=self.model_dir, device=self.device,
                                                graph=True, early_stop=3)
                return branch_graph_trainer.train()

        elif task == 'LNS':
            if method == 'lns_CL':
                trainer = CLTrainer(self.model_dir, self.data_dir, method, problem, self.device, gnn_type = 'gat', feature_set = 'feat2', loss = 'nt_xent', batch_size = 4)
                trainer.train(num_epochs=5, lr=0.001, warmstart=None, detect_anomalies=False, anneal_lr=False, give_up_after=100, decay_lr_after=20)
            if method == 'lns_IL':
                trainer_pre = ILTrainer_Pre(self.model_dir, self.data_dir, method, problem, self.device, batch_size = 4)
                trainer = ILTrainer(self.model_dir, self.data_dir, method, problem, self.device, gnn_type = 'gat', feature_set = 'feat2', loss = 'bce', batch_size = 4)
                trainer_pre.pre_model(lr_init = 0.0001, epoches = 30) 
                trainer.train(num_epochs=5, lr=0.001, warmstart=None, detect_anomalies=False, anneal_lr=False, give_up_after=100, decay_lr_after=20)
            if method == "lns_RL":
                gnnTrainer = GNNPolicyTrainer(self.device, method, problem, 'gnn',
                        'gnn_critic', noise_type = None,
                        normalize_returns = False,
                        normalize_observations = False, load_path = None,
                        model_dir = self.model_dir,
                        param_noise_adaption_interval = 30,
                        exploration_strategy = 'relpscost')

                gnnTrainer.train(self.data_dir, nb_epochs = 2, nb_rollout_steps = 2,
                        observation_range = (-np.inf, np.inf), 
                        action_range = (0.2, 0.8),
                        return_range = (-np.inf, np.inf),
                        reward_scale = 1.0, critic_l2_reg = 1e-2,
                        actor_lr = 1e-5,
                        critic_lr = 1e-5,
                        popart = False,
                        gamma = 0.99,  
                        clip_norm = None,
                        nb_train_steps = 50,
                        nb_eval_steps = 1000, #20 #50  
                        batch_size = 4, # per MPI worker  64 32  64   128   64  128 300
                        tau = 0.01, 
                        batch_sample = 1,
                        time_limit = 2,
                        eval_val = 0,
                        seed = 0)
            if method == "lns_GBDT":
                trainGNN(
                    self.device,
                    self.data_dir, 
                    self.model_dir,
                    method,
                    problem,
                    batch_size = 1,
                    learning_rate = 1e-3,
                    num_epochs =20,
                )
                trainGBDT(
                    self.device,
                    self.data_dir,
                    self.model_dir,
                    method,
                    problem
                )
        else:
            raise NotImplementedError
