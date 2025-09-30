import os
from os import path as osp
import pickle
import gzip
import torch
import numpy as np
from time import time as time
import random
import sys

import rl4mip.Trainer.Nodeselect_model.DSO.settings.consts as consts
import rl4mip.Trainer.Nodeselect_model.DSO.utils.logger as logger
import rl4mip.Trainer.Nodeselect_model.DSO.utils.utils as utils

import torch_geometric
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch.multiprocessing import Pool
from torch_geometric.data import Data
from torch import as_tensor

from rl4mip.Trainer.Nodeselect_model.DSO.utils.rl_algos import PPOAlgo
import rl4mip.Trainer.Nodeselect_model.DSO.dso_utils.expressions as expressions_module
from rl4mip.Trainer.Nodeselect_model.DSO.dso_utils.operators import Operators
from rl4mip.Trainer.Nodeselect_model.DSO.dso_utils.symbolic_agents import DSOAgent

import json
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .DataLoader import DsoSymbDataLoader
from pathlib import Path

class TrainDSOAgent(object):
    def __init__(self, 
                 seed=0,
                 total_iter=10,
                 batch_size=256,
                 data_batch_size=256,
                 eval_expression_num=48,
                 record_expression_num=16,
                 record_expression_freq=10,
                 early_stop=1000,
                 continue_train_path=None,
                 datapath='',
                 model_dir='',
                 device='cpu',
                 # env args
                 instance_kwargs={'instance_type': '', 'train_num': 100000, 'valid_num': 10000},
                 # expression
                 expression_kwargs={'math_list': 'simple', 'var_list': 'full'},
                 # agent
                 dso_agent_kwargs={'min_length': 4, 'max_length': 8, 'hidden_size': 128, 'num_layers': 2},
                 # rl_algo
                 rl_algo_kwargs={'class': 'PPOAlgo', 'kwargs': {'lr_actor': 5e-05, 'K_epochs': 8, 'entropy_coef': 0.05, 'entropy_gamma': 0.9, 'entropy_decrease': False, 'lr_decrease': False, 'decrease_period': 700}},
                 ):
        self.device = device
        self.model_dir = model_dir
        self.set_const_train()
        self.batch_size, self.data_batch_size, self.eval_expression_num, self.seed = batch_size, data_batch_size, eval_expression_num, seed
        self.early_stop, self.current_early_stop = early_stop, 0
        self.record_expression_num, self.record_expression_freq = record_expression_num, record_expression_freq
        self.instance_type = instance_kwargs["instance_type"]

        # self.total_iter = consts.ITER_DICT[self.instance_type] if total_iter is None else total_iter
        self.total_iter = total_iter

        # load datasets
        # self.train_data, self.valid_data, _ = get_all_dataset(datapath=datapath, device=device, **instance_kwargs)
        dataloader = DsoSymbDataLoader(datapath=datapath, device=device, **instance_kwargs)
        self.train_data, self.valid_data, _ = dataloader.dataloader()
        # expression
        self.operators = Operators(**expression_kwargs)
        # dso agent
        self.state_dict_dir, = logger.create_and_get_subdirs("state_dict")
        self.agent = DSOAgent(self.operators, **dso_agent_kwargs)
        if continue_train_path:
            logger.log(f"continue train from {continue_train_path} {consts.IMPORTANT_INFO_SUFFIX}")
            self.agent.load_state_dict(torch.load(continue_train_path))
        # rl algo
        self.rl_algo = PPOAlgo(agent=self.agent, **rl_algo_kwargs["kwargs"])
        # algo process variables
        self.train_iter = 0
        self.best_performance = - float("inf")
        self.best_writter = open(osp.join(logger.get_dir(), "best.txt"), "w")
        self.recorder = open(osp.join(logger.get_dir(), "all_expressions.txt"), "w")
        self.save_dir = osp.join(logger.get_dir(), "state_dict")
        
        self.datapath=datapath

    def set_const_train(self):
        DEVICE = torch.device(self.device)
        # DEVICE = torch.device("cpu")
        torch.set_default_device(DEVICE)
    def train(self):
        return self.process()



    def process(self):
        start_training_time = time()

        for self.train_iter in range(self.total_iter+1):
            if self.current_early_stop > self.early_stop:
                break

            # generate expressions
            sequences, all_lengths, log_probs, (all_counters_list, all_inputs_list) = self.agent.sample_sequence_eval(self.batch_size)
            expression_list = [expressions_module.Expression(sequence[:length], self.operators) for sequence, length in zip(sequences, all_lengths)]
            # train
            ensemble_expressions = expressions_module.EnsemBleExpression(expression_list)
            precisions = self.get_precision_iteratively(ensemble_expressions, self.train_data, self.data_batch_size, device=self.device)

            returns, indices = torch.topk(precisions, self.eval_expression_num, sorted=False)
            sequences, all_lengths, log_probs = sequences[indices], all_lengths[indices], log_probs[indices]
            all_counters_list, all_inputs_list = [all_counters[indices] for all_counters in all_counters_list], [all_inputs[indices] for all_inputs in all_inputs_list]

            index_useful = (torch.arange(sequences.shape[1], dtype=torch.long)[None, :] < all_lengths[:, None]).type(torch.float32)
            results_rl = self.rl_algo.train(sequences, all_lengths, log_probs, index_useful, (all_counters_list, all_inputs_list), returns=returns, train_iter=self.train_iter)

            ## tensorboard record
            results = {"train/batch_best_precision": precisions.max().item(),
                       "train/batch_topk_mean_precision": precisions[indices].mean(),
                       "train/batch_topk_var_precision": precisions[indices].std(),
                       "train/batch_all_mean_precision": precisions.mean(),
                       "train/train_iteration": self.train_iter,
                       "misc/cumulative_train_time": time() - start_training_time,
                       "misc/train_time_per_iteration": (time() - start_training_time)/(self.train_iter+1)
                       }
            results.update(results_rl)

            if self.train_iter % self.record_expression_freq == 0:
                _, where_to_valid = torch.topk(precisions, self.record_expression_num, sorted=True)
                expressions_to_valid = [expression_list[i.item()] for i in where_to_valid]
                ensemble_expressions_valid = expressions_module.EnsemBleExpression(expressions_to_valid)
                precisions_valid = self.get_precision_iteratively(ensemble_expressions_valid, self.valid_data)
                where_to_record = torch.where(precisions_valid > self.best_performance)[0]
                # print("where_to_record\n", where_to_record) #  tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15], device='cuda:1')
                if len(where_to_record) > 0:
                    self.current_early_stop = 0
                    pairs = [(expressions_to_valid[i], precisions_valid[i].item()) for i in where_to_record]
                    pairs.sort(key=lambda x: x[1])
                    self.best_performance = pairs[-1][1]
                    for (exp, value) in pairs:
                        best = f"iteration:{self.train_iter}_precision:{round(value, 4)}\t{exp.get_nlp()}\t{exp.get_expression()}\n"
                        best_value = value
                        best_expression = exp.get_expression()
                        self.best_writter.write(best)
                    logger.log(best)
                    # 记录最好的表达式
                    '''
                    {
                        "instance_type(eg:setcover)":{
                            "best_performance": 0.9752
                            "expression": "((((inputs[:,5] - inputs[:,25]) - inputs[:,23]) - inputs[:,4]) - inputs[:,24])"
                        },
                    }
                    '''
                    # best_expression_file_path = consts.BEST_EXPRESSION_FILE_PATH
                    model_path_directory = Path(os.path.join(self.model_dir, f'node_selection'))
                    model_path_directory.mkdir(parents=True, exist_ok=True)
                    best_expression_file_path = os.path.join(self.model_dir, "node_selection/policy_dso_symb.json")
                    # 如果文件不存在，则创建一个新的文件
                    best_expression_dict = {
                        self.instance_type:{
                            "best_performance": best_value,
                            "expression": best_expression
                        }
                    }
                    if not os.path.exists(best_expression_file_path) or os.path.getsize(best_expression_file_path) == 0:
                        os.makedirs(os.path.dirname(best_expression_file_path), exist_ok=True)
                        with open(best_expression_file_path, 'w') as f:
                            json.dump(best_expression_dict, f, indent=4)
                    else:
                        # 如果文件存在，并且有数据，读取文件中的数据
                        with open(best_expression_file_path, 'r') as f:
                            best_expression_dict = json.load(f)
                        # 如果已经存在这个问题的最好表达式
                        if self.instance_type in best_expression_dict:
                            cur_performance = best_expression_dict[self.instance_type]['best_performance']
                            if cur_performance < best_value:
                                best_expression_dict[self.instance_type]['expression'] = best_expression
                                best_expression_dict[self.instance_type]['best_performance'] = best_value
                        else:
                            cur_instance_dict = {}
                            cur_instance_dict['best_performance'] = best_value
                            cur_instance_dict['expression'] = best_expression
                            best_expression_dict[self.instance_type] = cur_instance_dict
                        # 打开文件并清空内容，同时写入字典数据
                        with open(best_expression_file_path, 'w') as f:
                            # 将字典数据写入文件并覆盖原有内容
                            json.dump(best_expression_dict, f, indent=4)

                    self.best_writter.flush()
                    os.fsync(self.best_writter.fileno())
                else:
                    self.current_early_stop += self.record_expression_freq
                results.update({
                    "valid/overall_best_precision": self.best_performance, 
                    "valid/valid_best_precision": precisions_valid.max().item(),
                    "valid/valid_all_mean_precision": precisions_valid.mean(),
                    "valid/valid_all_var_precision": precisions_valid.std(),
                    "valid/valid_iteration": self.train_iter
                })


                state_dict = self.agent.state_dict()
                state_dict_save_path = osp.join(self.save_dir, f"train_iter_{self.train_iter}_precision_{round(value, 4)}.pkl")
                torch.save(state_dict, state_dict_save_path)

            logger.logkvs_tb(results)
            logger.dumpkvs_tb()
        return best_expression_file_path


    def get_precision(self, model, batch, device):
        X, y_label, y_index = batch.x, batch.y, batch.y_batch
        pred_y = model(X, device=device, train_mode=False)
        real_label = pred_y > 0
        comparison = real_label == y_label
        real_label = comparison.sum(dim=1)
        return real_label

    @torch.no_grad()
    def get_precision_iteratively(self, model, data, partial_sample=None, device='cpu'):
        scores_sum, data_sum = 0, 0
        if partial_sample is None:
            partial_sample = len(data)
        for batch in data:
            data_num = batch.y.shape[0]
            batch = batch.to(device)
            batch_labels = self.get_precision(model, batch, device)
            if data_sum >= partial_sample:
                break
        result = batch_labels / data_num
        return result