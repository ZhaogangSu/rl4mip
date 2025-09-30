import numpy as np
import datetime
import os
import pathlib
import torch
import torch.nn.functional as F

from .DataLoader import GNNdataLoader
from ml4co.Trainer.Branch_model.GNN import GNNPolicy


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

class GNNBranchTrainer:
    """"""
    def __init__(self, problem, model_dir, datapath, 
                    batch_size, pretrain_batch_size, valid_batch_size,
                    device, entropy_bonus = 0.0, top_k = [1,3,5,10], seed=0):
        
        self.dataloader = GNNdataLoader(problem, datapath)
        self.pretrain_loader = self.dataloader.loadpretraind(pretrain_batch_size)
        self.valid_loader = self.dataloader.loadvalid(valid_batch_size)
        self.batch_size = batch_size
        self.seed = seed
        self.top_k = top_k
        self.entropy_bonus = entropy_bonus
        self.device = device
        self.problem = problem
        self.model_path = os.path.join(model_dir, "gnn_policy")
        os.makedirs(self.model_path, exist_ok=True)

    def pretrain(self, policy):
        policy.pre_train_init()
        i = 0
        while True:
            for batch in self.pretrain_loader:
                batch.to(self.device)
                if not policy.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features):
                    break

            if policy.pre_train_next() is None:
                break
            i += 1
        return i
    
    def process(self, policy, data_loader, optimizer=None):
        mean_loss = 0
        mean_kacc = np.zeros(len(self.top_k))
        mean_entropy = 0

        n_samples_processed = 0
        with torch.set_grad_enabled(optimizer is not None):
            for batch in data_loader:
                batch = batch.to(self.device)
                logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
                cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
                entropy = (-F.softmax(logits, dim=-1)*F.log_softmax(logits, dim=-1)).sum(-1).mean()
                loss = cross_entropy_loss - self.entropy_bonus*entropy

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
                true_bestscore = true_scores.max(dim=-1, keepdims=True).values

                kacc = []
                for k in self.top_k:
                    if logits.size()[-1] < k:
                        kacc.append(1.0)
                        continue
                    pred_top_k = logits.topk(k).indices
                    pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
                    accuracy = (pred_top_k_true_scores == true_bestscore).any(dim=-1).float().mean().item()
                    kacc.append(accuracy)

                kacc = np.asarray(kacc)
                mean_loss += cross_entropy_loss.item() * batch.num_graphs
                mean_entropy += entropy.item() * batch.num_graphs
                mean_kacc += kacc * batch.num_graphs
                n_samples_processed += batch.num_graphs

        mean_loss /= n_samples_processed
        mean_kacc /= n_samples_processed
        mean_entropy /= n_samples_processed
        return mean_loss, mean_kacc, mean_entropy


    def train(self, epochs=1000, initial_lr=1e-3, patience=15, early_stopping=30):
        ### log ###
        lr = initial_lr
        log(f"max_epochs: {epochs}")
        log(f"batch_size: {self.batch_size}")
        log(f"lr: {initial_lr}")
        log(f"patience : {patience}")
        log(f"early_stopping : {early_stopping}")
        log(f"entropy bonus: {self.entropy_bonus}")
        log(f"top_k: {self.top_k}")
        log(f"device: {self.device}")
        log(f"seed {self.seed}")

        ### init ###
        policy = GNNPolicy().to(self.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience)
        best_loss = np.inf

        saved_model_dir = self.model_path
        saved_model = os.path.join(saved_model_dir, f'{self.problem}.pkl')

        for epoch in range(epochs + 1):
            log(f"EPOCH {epoch}...")
            if epoch == 0:
                n = self.pretrain(policy)
                log(f"PRETRAINED {n} LAYERS")
            else:
                train_loader = self.dataloader.loadepochtrain(self.batch_size, self.seed)
                train_loss, train_kacc, entropy = self.process(policy, train_loader, optimizer)
                log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(self.top_k, train_kacc)]))

            # test
            valid_loss, valid_kacc, entropy = self.process(policy, self.valid_loader, None)
            log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(self.top_k, valid_kacc)]))

            if valid_loss < best_loss:
                plateau_count = 0
                best_loss = valid_loss
                torch.save(policy.state_dict(), saved_model)
                log(f"  best model so far")

            else:
                plateau_count += 1
                if plateau_count % early_stopping == 0:
                    log(f"  {plateau_count} epochs without improvement, early stopping")
                    break
                if plateau_count % patience == 0:
                    lr *= 0.2
                    log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}")

            scheduler.step(valid_loss)

        policy.load_state_dict(torch.load(saved_model))
        valid_loss, valid_kacc, entropy = self.process(policy, self.valid_loader, None)
        log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(self.top_k, valid_kacc)]))

        return saved_model




        




        
