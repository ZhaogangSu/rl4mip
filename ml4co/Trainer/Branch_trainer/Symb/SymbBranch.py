from .utils import TrainDSOAgent
import os

class SymbBranchTrainer:
    def __init__(self, problem, datapath, model_dir, device, graph=False, seed=0,
                        batch_size=1024, data_batch_size=1000,
                        train_num=1000, valid_num=400,
                        eval_expression_num=48,
                        record_expression_num=16,
                        record_expression_freq=10,
                        early_stop=400):
        self.agent = TrainDSOAgent(problem, datapath, model_dir, device, graph,
                                    seed=seed, batch_size=batch_size, data_batch_size=data_batch_size,
                                    train_num=train_num, valid_num=valid_num,
                                    eval_expression_num=eval_expression_num,
                                    record_expression_num=record_expression_num,
                                    record_expression_freq=record_expression_freq,
                                    early_stop=early_stop)
        self.problem = problem
        self.model_dir = model_dir
        self.graph = graph

    def train(self,epochs):
        if not self.graph:
            logdir_record_path = self.agent.process(epochs)
            del self.agent
            expression_save_path = os.path.join(self.model_dir, "symb_policy", f"{self.problem}.txt")
            exp_list = []
            with open(expression_save_path, "a") as expression_save_file, open(logdir_record_path, "r") as logdir_record_file:
                lines = logdir_record_file.readlines()
                for line in reversed(lines):
                    assert line[:9] == "iteration"
                    exps = line.split("\t")
                    exps = [x.strip() for x in exps if "inputs[" in x]
                    exp_list.extend(exps)
                    if len(exp_list) >= 1:
                        break
                expression_save_file.write(exp_list[0])

        else:
            logdir_record_path = self.agent.process_graph(epochs)
            del self.agent
            expression_save_path = os.path.join(self.model_dir, "graph_policy", f"{self.problem}.txt")
            exp_list = []
            with open(expression_save_path, "a") as expression_save_file, open(logdir_record_path, "r") as logdir_record_file:
                lines = logdir_record_file.readlines()
                for line in reversed(lines):
                    assert line[:9] == "iteration"
                    exps = line.split("\t")
                    exps = [x.strip() for x in exps if "variable[" in x]
                    exp_list.extend(exps)
                    if len(exp_list) >= 1:
                        break
                expression_save_file.write(exp_list[0])

        return expression_save_path
                    
        


