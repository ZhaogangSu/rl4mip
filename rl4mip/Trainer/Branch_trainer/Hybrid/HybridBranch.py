import numpy as np
import torch
import os
import sys
import importlib

from .DataLoader import HybriddataLoader
from .utils import log, _loss_fn, _distillation_loss, _compute_root_loss

class HybridBranchTrainer:
    """"""
    def __init__(self, problem, datapath, device, model_dir, accum_steps=1,
                    epoch_size=312, batch_size=32, pretrain_batch_size=128, valid_batch_size=128,
                    no_e2e=True, distilled=True, T=2, alpha=0.9, 
                    AT='', beta_at=0, root_cands_separation=False, 
                    l2=0.0, top_k = [1,3,5,10], seed=0):
        
        self.dataloader = HybriddataLoader(problem, datapath)
        self.pretrain_loader = self.dataloader.loadpretraind(pretrain_batch_size)
        self.valid_loader = self.dataloader.loadvalid(valid_batch_size)
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.accum_steps = accum_steps

        self.seed = seed
        self.top_k = top_k
        self.device = device
        self.model_path = os.path.join(model_dir, 'hybrid_policy', problem)
        os.makedirs(self.model_path, exist_ok=True)

        self.no_e2e = no_e2e
        self.distilled = distilled
        self.T = T
        self.alpha = alpha
        self.AT = AT
        self.beta_at = beta_at
        self.l2 = l2
        self.root_cands_separation = root_cands_separation

    def pretrain(self, model):
        model.pre_train_init()
        i = 0
        while True:
            for batch in self.pretrain_loader:
                root_g, node_g, node_attr = [map(lambda x:x if x is None else x.to(self.device) , y) for y in batch]
                root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, *_ = root_g
                g_c, g_ei, g_ev, g_v, g_n_cs, g_n_vs, candss = node_g
                cand_features, n_cands, best_cands, cand_scores, weights = node_attr
                batched_states = (root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, candss, cand_features, None)
                if not model.pre_train(batched_states):
                    break

            res = model.pre_train_next()
            if res is None:
                break
            else:
                layer = res
            i += 1

        return i
    
    def process(self, model, teacher, data_loader, optimizer=None):
        mean_loss = 0
        mean_kacc = np.zeros(len(self.top_k))

        n_samples_processed = 0
        accum_iter = 0
        for batch in data_loader:
            root_g, node_g, node_attr = [map(lambda x:x if x is None else x.to(self.device) , y) for y in batch]
            root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, root_cands, root_n_cands = root_g
            node_c, node_ei, node_ev, node_v, node_n_cs, node_n_vs, candss = node_g
            cand_features, n_cands, best_cands, cand_scores, weights  = node_attr
            cands_root_v = None

            # use teacher
            with torch.no_grad():
                if teacher is not None:
                    if self.no_e2e:
                        root_v, _ = teacher((root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs))
                        cands_root_v = root_v[candss]

                    # KD - get soft targets
                    if self.distilled:
                        _, soft_targets = teacher((node_c, node_ei, node_ev, node_v, node_n_cs, node_n_vs))
                        soft_targets = torch.unsqueeze(torch.gather(input=torch.squeeze(soft_targets, 0), dim=0, index=candss), 0)
                        soft_targets = model.pad_output(soft_targets, n_cands)  # apply padding now

            batched_states = (root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, candss, cand_features, cands_root_v)
            batch_size = n_cands.shape[0]
            weights /= batch_size # sum loss

            if optimizer:
                optimizer.zero_grad()
                var_feats, logits, film_parameters = model(batched_states)  # eval mode
                logits = model.pad_output(logits, n_cands)  # apply padding now

                # node loss
                if self.distilled:
                    loss = _distillation_loss(logits, soft_targets, best_cands, weights, self.T, self.alpha)
                else:
                    loss = _loss_fn(logits, best_cands, weights)

                # AT loss
                if self.AT != "":
                    loss  += self.beta_at * _compute_root_loss(self.AT, model, var_feats, root_n_vs, root_cands, root_n_cands, batch_size, self.root_cands_separation)

                # regularization
                if (
                    self.l2 > 0
                    and film_parameters is not None
                ):
                    beta_norm = (1-film_parameters[:, :, 0]).norm()
                    gamma_norm = film_parameters[:, :, 1].norm()
                    loss += self.l2 * (beta_norm + gamma_norm)

                loss.backward()
                accum_iter += 1
                if accum_iter % self.accum_steps == 0:
                    optimizer.step()
                    accum_iter = 0
            else:
                with torch.no_grad():
                    var_feats, logits, film_parameters = model(batched_states)  # eval mode
                    logits = model.pad_output(logits, n_cands)  # apply padding now

                    # node loss
                    if self.distilled:
                        loss = _distillation_loss(logits, soft_targets, best_cands, weights, self.T, self.alpha)
                    else:
                        loss = _loss_fn(logits, best_cands, weights)

                    # AT loss
                    if self.AT != "":
                        loss  += self.beta_at * _compute_root_loss(self.AT, model, var_feats, root_n_vs, root_cands, root_n_cands, batch_size, self.root_cands_separation)

                    # regularization
                    if (
                        self.l2 > 0
                        and film_parameters is not None
                    ):
                        beta_norm = (1-film_parameters[:, :, 0]).norm()
                        gamma_norm = film_parameters[:, :, 1].norm()
                        loss += self.l2 * (beta_norm + gamma_norm)

            true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
            true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
            true_scores = true_scores.cpu().numpy()
            true_bestscore = true_bestscore.cpu().numpy()

            kacc = []
            for k in self.top_k:
                pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
                pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
                kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
            kacc = np.asarray(kacc)

            mean_loss += loss.detach_().item() * batch_size
            mean_kacc += kacc * batch_size
            n_samples_processed += batch_size

        mean_loss /= n_samples_processed
        mean_kacc /= n_samples_processed

        return mean_loss, mean_kacc


    def train(self, model_name='film', teacher_model='gnn_policy',
                    epochs=1000, lr=1e-3, patience=15, early_stopping=30):
        
        ### log ###
        log(f"max_epochs: {epochs}")
        log(f"epoch_size: {self.epoch_size}")
        log(f"batch_size: {self.batch_size}")
        log(f"lr: {lr}")
        log(f"patience : {patience}")
        log(f"early_stopping : {early_stopping}")
        log(f"top_k: {self.top_k}")
        log(f"device: {self.device}")
        log(f"seed {self.seed}")
        log(f"e2e: {not self.no_e2e}")
        log(f"KD: {self.distilled}")
        log(f"AT: {self.AT} beta={self.beta_at}")
        log(f"l2: {self.l2}")

        rng = np.random.RandomState(self.seed)
        torch.manual_seed(rng.randint(np.iinfo(int).max))

        if (model_name in ['concat', 'film'] and self.no_e2e):
            model_name = f"{model_name}-pre"

        model_file = model_name
        if self.distilled:
            model_file = f"{model_file}_distilled"

        if self.AT != "":
            model_file = f"{model_file}_{self.AT}_{self.beta_at}"

        if self.l2 > 0:
            model_file = f"{model_file}_l2_{self.l2}"

        ### model loading ###
        sys.path.insert(0, os.path.abspath(f'rl4mip/Trainer/Branch_model/hybrid_model/{model_name}'))
        import model
        importlib.reload(model)
        distilled_model = model.Policy()
        del sys.path[0]
        distilled_model.to(self.device)

        ### teacher model loading ###
        teacher = None
        if (self.distilled or self.no_e2e):
            sys.path.insert(0, os.path.abspath(f'rl4mip/Trainer/Branch_model/hybrid_model/{teacher_model}'))
            import model
            importlib.reload(model)
            teacher = model.GCNPolicy()
            del sys.path[0]
            teacher.restore_state(f"{self.model_path}/gnn_params.pkl")
            teacher.to(self.device)
            teacher.eval()

        model = distilled_model

        ### training loop ###
        saved_model = os.path.join(self.model_path, f'{model_file}_params.pkl')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience)
        best_loss = np.inf
        for epoch in range(epochs + 1):
            log(f"EPOCH {epoch}...")

            if (epoch == 0 and not self.no_e2e):
                n = self.pretrain(model)
                log(f"PRETRAINED {n} LAYERS")
            else:
                # bugfix: tensorflow's shuffle() seems broken...
                train_loader = self.dataloader.loadepochtrain(self.batch_size, self.epoch_size, self.accum_steps, self.seed)
                train_loss, train_kacc = self.process(model, teacher, train_loader, optimizer)
                log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(self.top_k, train_kacc)]))

            # TEST
            valid_loss, valid_kacc = self.process(model, teacher, self.valid_loader, None)
            log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(self.top_k, valid_kacc)]))

            if valid_loss < best_loss:
                plateau_count = 0
                best_loss = valid_loss
                model.save_state(saved_model)
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

        model.restore_state(saved_model)
        valid_loss, valid_kacc = self.process(model, teacher, self.valid_loader, None)
        log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(self.top_k, valid_kacc)]))

        return model_name, saved_model
        


        
        
        

