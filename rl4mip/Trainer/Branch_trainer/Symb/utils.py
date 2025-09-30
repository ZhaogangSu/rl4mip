from rl4mip.Trainer.Branch_model.symb_model.rl_algos import PPOAlgo
import rl4mip.Trainer.Branch_model.symb_model.expressions as expressions_module
import rl4mip.Trainer.Branch_model.symb_model.expressions_graph as expressions_module_graph
from rl4mip.Trainer.Branch_model.symb_model.operators import Operators
from rl4mip.Trainer.Branch_model.symb_model.operators_graph import Operators_Graph
from rl4mip.Trainer.Branch_model.symb_model.symbolic_agents import DSOAgent
from rl4mip.Trainer.Branch_model.symb_model.graph_agents import TransformerDSOAgent
from .DataLoader import get_all_dataset
import datetime
import os
import torch
import numpy as np
from time import time as time
from torch_scatter import scatter_max

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def get_precision(model, batch):
    X, y_label, y_index = batch.x, batch.y, batch.y_batch
    pred_y = model(X, train_mode=False)
    _, where_max = scatter_max(pred_y, y_index)
    where_illegal = (where_max == len(y_label))
    where_max[where_illegal] = 0
    real_label = y_label[where_max]
    real_label[where_illegal] = False
    return real_label

def get_batch_score_precision(model, batch):
    pred_y = model(batch, train_mode=False)
    _, where_max = scatter_max(pred_y, batch.y_cand_mask_batch)
    where_max_illegal = where_max==len(batch.y_cand_label)
    where_max[where_max_illegal] = 0
    real_label = batch.y_cand_label[where_max]
    real_label[where_max_illegal] = False
    return real_label

@torch.no_grad()
def get_precision_iteratively(model, data, partial_sample=None, device='', graph=False):

    scores_sum, data_sum = 0, 0
    if partial_sample is None:
        partial_sample = len(data)
    for batch in data:
        batch = batch.to(device)
        if not graph:
            batch_labels = get_precision(model, batch)
        else:
            batch_labels = get_batch_score_precision(model, batch)
        scores_sum += batch_labels.sum(dim=-1)
        data_sum += len(batch)
        if data_sum >= partial_sample:
            break
    result = scores_sum / data_sum
    return result


class TrainDSOAgent(object):
    def __init__(self, problem, datapath, model_dir, device,
                 graph=False,
                 seed=0,
                 batch_size=1024,
                 data_batch_size=1000,

                 train_num=1000,
                 valid_num=400,
                 
                 eval_expression_num=48,
                 record_expression_num=16,
                 record_expression_freq=10,

                 early_stop=10,
                 ):
        self.device = device
        torch.set_default_device(torch.device(self.device))
        self.batch_size, self.data_batch_size, self.eval_expression_num, self.seed = batch_size, data_batch_size, eval_expression_num, seed
        self.early_stop, self.current_early_stop = early_stop, 0
        self.record_expression_num, self.record_expression_freq = record_expression_num, record_expression_freq
        self.graph = graph

        # load datasets
        self.train_data, self.valid_data, _ = get_all_dataset(problem, datapath, self.device, self.graph, train_num, valid_num)

        # expression
        if not self.graph:
            self.operators = Operators(math_list="simple", var_list="full")
        else:
            self.operators = Operators_Graph(math_list='simple', var_list='graph', scatter_max_degree=2)

        # dso agent
        if not self.graph:
            self.agent = DSOAgent(self.operators, min_length=4, max_length=64, hidden_size=128, num_layers=2)
        else:
            self.agent = TransformerDSOAgent(self.operators)

        # rl algo
        self.rl_algo = PPOAlgo(agent=self.agent, lr_actor=5e-5, K_epochs=8, 
                                entropy_coef=0.05, entropy_gamma=0.9, 
                                entropy_decrease=False, lr_decrease=False,
                                decrease_period=700)

        # algo process variables
        self.train_iter = 0
        self.best_performance = - float("inf")
        if not self.graph:
            os.makedirs(os.path.join(model_dir, "symb_policy"), exist_ok=True)
            self.tmp_writter = open(os.path.join(model_dir, "symb_policy", f"{problem}_tmp.txt"), "w")
        else:
            os.makedirs(os.path.join(model_dir, "graph_policy"), exist_ok=True)
            self.tmp_writter = open(os.path.join(model_dir, "graph_policy", f"{problem}_tmp.txt"), "w")
        

    def process(self,epoch):
        start_training_time = time()
        for self.train_iter in range(epoch):
            if self.current_early_stop > self.early_stop:
                break

            # generate expressions
            sequences, all_lengths, log_probs, (all_counters_list, all_inputs_list) = self.agent.sample_sequence_eval(self.batch_size)
            expression_list = [expressions_module.Expression(sequence[:length], self.operators) for sequence, length in zip(sequences, all_lengths)]

            # train
            ensemble_expressions = expressions_module.EnsemBleExpression(expression_list)
            precisions = get_precision_iteratively(ensemble_expressions, self.train_data, self.data_batch_size, device=self.device)

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
                precisions_valid = get_precision_iteratively(ensemble_expressions_valid, self.valid_data, device=self.device)
                where_to_record = torch.where(precisions_valid > self.best_performance)[0]
                if len(where_to_record) > 0:
                    self.current_early_stop = 0
                    pairs = [(expressions_to_valid[i], precisions_valid[i].item()) for i in where_to_record]
                    pairs.sort(key=lambda x: x[1])
                    self.best_performance = pairs[-1][1]
                    for (exp, value) in pairs:
                        best = f"iteration:{self.train_iter}_precision:{round(value, 4)}\t{exp.get_nlp()}\t{exp.get_expression()}\n"
                        self.tmp_writter.write(best)
                    log(best)
                    self.tmp_writter.flush()
                    os.fsync(self.tmp_writter.fileno())
                else:
                    self.current_early_stop += self.record_expression_freq
                results.update({
                    "valid/overall_best_precision": self.best_performance, 
                    "valid/valid_best_precision": precisions_valid.max().item(),
                    "valid/valid_all_mean_precision": precisions_valid.mean(),
                    "valid/valid_all_var_precision": precisions_valid.std(),
                    "valid/valid_iteration": self.train_iter
                })

            log(results)

        return self.tmp_writter.name

    
    def process_graph(self,epoch):
        start_time = time()
        for self.train_iter in range(epoch):
            if self.current_early_stop > self.early_stop:
                break
            iter_start_time = time()

            sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, silbing_pairs, silbing_length) = self.agent.sample_sequence_eval(self.batch_size)
            expression_list = [expressions_module_graph.Expression(sequence[1:length+1], scatter_degree_now[:length], self.operators) for sequence, length, scatter_degree_now in zip(sequences, all_lengths, scatter_degree)]

            expression_generation_time = time() - iter_start_time

            eval_expression_start_time = time()

            # train
            ensemble_expressions = expressions_module_graph.EnsemBleExpression(expression_list)
            precisions = get_precision_iteratively(ensemble_expressions, self.train_data, self.data_batch_size, device=self.device, graph=True)

            eval_expression_time = time() - eval_expression_start_time

            rl_start_time = time()
            returns, indices = torch.topk(precisions, self.eval_expression_num, sorted=False)

            sequences, all_lengths, log_probs = sequences[indices], all_lengths[indices], log_probs[indices]
            scatter_degree, all_counters_list, scatter_parent_where_seq = scatter_degree[indices],\
                                                                        [all_counters[indices] for all_counters in all_counters_list],\
                                                                        scatter_parent_where_seq[indices]
            parent_useful_index = torch.any(parent_child_pairs[:,0][:, None] == indices[None,:], dim=1)
            parent_child_pairs = parent_child_pairs[parent_useful_index]
            parent_useful_cumsum = torch.cumsum(parent_useful_index.long(),dim=0)
            parent_child_length[1:] = parent_useful_cumsum[parent_child_length[1:]-1]
            parent_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
            parent_new_index0[indices] = torch.arange(len(indices))
            parent_child_pairs[:, 0] = parent_new_index0[parent_child_pairs[:, 0]]


            silbling_useful_index = torch.any(silbing_pairs[:,0][:, None] == indices[None,:], dim=1)
            silbing_pairs = silbing_pairs[silbling_useful_index]
            silbing_useful_cumsum = torch.cumsum(silbling_useful_index.long(), dim=0)
            where_start_positive = torch.where(silbing_length > 0)[0][0]
            silbing_length[where_start_positive:] = silbing_useful_cumsum[silbing_length[where_start_positive:]-1]
            silbing_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
            silbing_new_index0[indices] = torch.arange(len(indices))
            silbing_pairs[:, 0] = silbing_new_index0[silbing_pairs[:, 0]]

            assert (silbing_pairs[:, 0].min() == parent_child_pairs[:, 0].min() == 0) and (silbing_pairs[:, 0].max() == parent_child_pairs[:, 0].max() ==  len(indices) - 1)

            index_useful = (torch.arange(sequences.shape[1]-1, dtype=torch.long)[None, :] < all_lengths[:, None]).type(torch.float32)

            results_rl = self.rl_algo.train(sequences, all_lengths, log_probs, index_useful, (scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, silbing_pairs, silbing_length), returns=returns, train_iter=self.train_iter)


            iter_end_time = time()
            rl_time = iter_end_time - rl_start_time
            iter_time = iter_end_time - iter_start_time

            ## tensorboard record
            total_time = iter_end_time - start_time
            results = {"train/batch_best_loss": returns.max().item(),
                       "train/batch_topk_mean_loss": returns.mean(),
                       "train/batch_topk_var_loss": returns.std(),
                       "train/batch_all_mean_loss": precisions.mean(),
                       "train/batch_all_var_loss": precisions.std(),

                       "train/train_iteration": self.train_iter,
                       "train/iter_time": iter_time,
                       "train/iter_time_generation": expression_generation_time,
                       "train/iter_time_evaluation": eval_expression_time,
                       "train/iter_time_rl": rl_time,
                       "train/total_time": total_time
                       }
            results.update(results_rl)

            ## save expressions and models
            if self.train_iter % self.record_expression_freq == 0:
                _, where_to_valid = torch.topk(precisions, self.record_expression_num, sorted=True)

                expressions_to_valid = [expression_list[i.item()] for i in where_to_valid]
                ensemble_expressions_valid = expressions_module_graph.EnsemBleExpression(expressions_to_valid)

                loss_valid = get_precision_iteratively(ensemble_expressions_valid, self.valid_data, device=self.device, graph=True)
                precisions_valid = get_precision_iteratively(ensemble_expressions_valid, self.valid_data, device=self.device, graph=True)

                where_to_record = torch.where(loss_valid > self.best_performance)[0]
                if len(where_to_record) > 0:
                    self.current_early_stop = 0
                    pairs = [(expressions_to_valid[i], loss_valid[i].item(), precisions_valid[i]) for i in where_to_record]
                    pairs.sort(key=lambda x: x[1])
                    self.best_performance = pairs[-1][1]
                    for (exp, value, precision_value) in pairs:
                        best = f"iteration:{self.train_iter}_loss:{round(value, 4)}_precision:{round(precision_value.item(), 4)}\t{exp.get_nlp()}\t{exp.get_expression()}\n"
                        self.tmp_writter.write(best)
                    log(best)
                    self.tmp_writter.flush()
                    os.fsync(self.tmp_writter.fileno())
                else:
                    self.current_early_stop += self.record_expression_freq
                results.update({
                    "valid/overall_best_loss": self.best_performance,

                    "valid/valid_best_loss": loss_valid.max().item(),
                    "valid/valid_all_mean_loss": loss_valid.mean(),
                    "valid/valid_all_var_loss": loss_valid.std(),

                    "valid/valid_best_loss_precision": precisions_valid[torch.argmax(loss_valid)].item(),
                    "valid/valid_best_precision": precisions_valid.max().item(),
                    "valid/valid_all_mean_precision": precisions_valid.mean(),
                    "valid/valid_all_var_precision": precisions_valid.std(),

                    "valid/valid_iteration": self.train_iter,
                })
            
            log(results)

        return self.tmp_writter.name
    
