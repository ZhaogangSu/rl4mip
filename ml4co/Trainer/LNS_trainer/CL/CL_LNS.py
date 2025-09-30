import torch
import torch.nn.functional as F
import time
import math
import os
import glob
import sys
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning import losses
from torchmetrics.functional import auroc
from tensorboardX import SummaryWriter as SummaryWriter
from ml4co.Trainer.LNS_model.CL_model.gnn_policy import GNNPolicy
from ml4co.Trainer.LNS_model.CL_model.losses import LogScoreLoss, LinearScoreLoss
from ml4co.DataCollector.LNS_data.CL_data.utils import augment_variable_features_with_dynamic_ones
from ml4co.DataCollector.LNS_data.CL_data.utils import pad_tensor, multi_hot_encoding
from ml4co.DataCollector.LNS_data.CL_data import bipartite_graph_loader as bgl


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class CLTrainer():
    def __init__(self, dir_model, dir_path, method, problem, device, gnn_type, feature_set, loss, 
                    batch_size=8, checkpoint_every=40):
        
        databases_train = glob.glob(os.path.join(dir_path, "samples", method, problem, "train/*.db"))
        databases_valid = glob.glob(os.path.join(dir_path, "samples", method, problem, "valid/*.db"))
        print(databases_train)
        print(databases_valid)
        print(f"{len(databases_train)} train databases")
        print(f"{len(databases_valid)} validation databases")

        train_db = []
        valid_db = []
        for i, database in enumerate(databases_train):
            try:
                loader = bgl.BipartiteGraphLoader(database, shuffle=True)
            except:
                continue
            if loader.num_examples() == 0:
                continue
            train_db.append(database)

        for i, database in enumerate(databases_valid):
            try:
                loader = bgl.BipartiteGraphLoader(database, shuffle=True)
            except:
                continue
            if loader.num_examples() == 0:
                continue
            valid_db.append(database)        

        train_dbs = "+".join(train_db)
        valid_dbs = "+".join(valid_db)
        print(train_dbs)
        self.train_loader = bgl.BipartiteGraphLoader(train_dbs, shuffle=True, first_k=None)
        self.valid_loader = bgl.BipartiteGraphLoader(valid_dbs, shuffle=False)
        
        print(f"Training on {self.train_loader.num_examples()} examples")
        print(f"Evaluating on {self.valid_loader.num_examples()} examples")

        self.problem = problem
        self.gnn_type = gnn_type
        self.device = device
        self.loss = loss

        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every

        self.experiment = feature_set + '_' + gnn_type
        save_to_folder = os.path.join(dir_model, method, problem, f"model_{problem}_{feature_set}_{loss}/")
        self.checkpoint = save_to_folder + "neural_LNS_" + problem + "_" + self.experiment + ".pt"
        self.tensorboard = save_to_folder + "neural_LNS_" + problem + "_" + self.experiment + ".tb"

    def load_policy_from_checkpoint(self, warmstart):

        policy = GNNPolicy(self.gnn_type)
        
        try:
            ckpt = torch.load(warmstart, map_location=self.device)
            try_again = False
        except Exception as e:
            print("Checkpoint " + self.checkpoint + " not found, bailing out: " + str(e))
            sys.exit(1)
        
        policy.load_state_dict(ckpt.state_dict())
        
        print("Loaded checkpoint")
        print(f"Will run evaluation on {self.device} device", flush=True)

        return policy

    def process(self, policy, data_loader, optimizer=None):
        """
        This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
        """
        prefix = "Train" if optimizer else "Eval"

        log_score_loss_function = LogScoreLoss().to(self.device)
        linear_score_loss_function = LinearScoreLoss().to(self.device)
        bce_loss_function = torch.nn.BCEWithLogitsLoss(reduction="none").to(self.device)
        infoNCE_loss_function = losses.NTXentLoss(temperature=0.07,distance=DotProductSimilarity()).to(self.device)

        if self.loss == "linear_score":
            loss_function = linear_score_loss_function
        elif self.loss == "log_score":
            loss_function = log_score_loss_function
        else:
            loss_function = bce_loss_function

        mean_loss = 0.0
        mean_acc = 0.0
        mean_auc = 0.0

        mean_offby = 0.0

        top_k = [1, 3, 5, 10]
        k_acc = [0.0, 0.0, 0.0, 0.0]

        n_iters = 0
        n_samples_processed = 0
        n_positive_samples = 0
        n_negative_samples = 0

        start = time.time()
        n_samples_previously_processed = 0

        history_window_size = 3

        with torch.set_grad_enabled(optimizer is not None):
            for batch in data_loader:
                batch = batch.to(self.device)

                # TO DO: Fix the dataset instead
                if torch.isnan(batch.candidate_scores).any():
                    print("Skipping batch with NaN scores")
                    continue

                batch = augment_variable_features_with_dynamic_ones(batch, self.device, self.experiment, self.problem)
                
                # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
                try:
                    logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                except RuntimeError as e:
                    print("Skipping batch due to error: " + str(e))
                    continue

                # Index the results by the candidates, and split and pad them
                pred_scores = pad_tensor(logits, batch.nb_candidates, normalize=False)
                true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates, normalize=False)

                assert not torch.isnan(pred_scores).any()
                assert not torch.isnan(true_scores).any()
                
                if self.loss == "cross_entropy":
                    # Compute the usual cross-entropy classification loss
                    loss = F.cross_entropy(pred_scores, batch.candidate_choices)
                elif self.loss == "bce":
                    multi_hot_labels = multi_hot_encoding(true_scores)
                    raw_loss = bce_loss_function(pred_scores, multi_hot_labels)
                    batch_loss = torch.mean(raw_loss, 1)
                    loss_sum = torch.sum(torch.mul(batch_loss, batch.batch_weight))
                    loss = torch.div(loss_sum, torch.sum(batch.batch_weight))
                    
                elif self.loss == "nt_xent":
                    batch_size = pred_scores.shape[0]
                    multi_hot_labels = multi_hot_encoding(true_scores)
                    embeddings = torch.sigmoid(pred_scores)
                    anchor_positive = []
                    anchor_negative = []
                    positive_idx = []
                    negative_idx = []
                    total_sample = batch_size
                    
                    for i in range(batch_size):
                        if batch.batch_weight[i].item() == 1:
                            #embed()
                            #anchor.append(i)
                            if len(batch.info["positive_samples"][i]) == 0: #due to unknown bugs for SC
                                #embed()
                                continue
                            ground_truth_improvement = max(batch.info["positive_labels"][i])
                            for j in range(len(batch.info["positive_samples"][i])):
                                improvement_j = batch.info["positive_labels"][i][j]
                                if improvement_j >= ground_truth_improvement * 0.5:
                                    anchor_positive.append(i)
                                    positive_idx.append(total_sample)
                                    embeddings = torch.cat([embeddings, torch.tensor([batch.info["positive_samples"][i][j]]).to(self.device)])
                                    total_sample += 1
                                    n_positive_samples += 1
                            for j in range(len(batch.info["negative_samples"][i])):
                                improvement_j = batch.info["negative_labels"][i][j]
                                if improvement_j <= ground_truth_improvement * 0.05:
                                    anchor_negative.append(i)
                                    negative_idx.append(total_sample)
                                    embeddings = torch.cat([embeddings, torch.tensor([batch.info["negative_samples"][i][j]]).to(self.device)])
                                    total_sample += 1
                                    n_negative_samples += 1

                    triplets = (torch.tensor(anchor_positive).to(self.device), torch.tensor(positive_idx).to(self.device), torch.tensor(anchor_negative).to(self.device), torch.tensor(negative_idx).to(self.device))
                    loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)
                else:
                    # use the log or linear score loss
                    normalized_scores = normalize_tensor(batch.candidate_scores)
                    loss = loss_function(logits[batch.candidates], normalized_scores)
                
                if  math.isnan(loss.item()):
                    continue

                assert not math.isnan(loss.item())
                if not (loss.item() >= 0 or  torch.sum(batch.batch_weight).item() == 0):
                    print("Error")

                assert loss.item() >= 0 or  torch.sum(batch.batch_weight).item() == 0, f"loss = {loss.item()}, #samples = {torch.sum(batch.batch_weight).item()}"
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        
                mean_loss += loss.item() * torch.sum(batch.batch_weight).item()
                n_samples_processed += torch.sum(batch.batch_weight).item()# batch.num_graphs
                n_iters += 1
                
                for i in range(multi_hot_labels.shape[0]):
                    if batch.batch_weight[i].item() == 0:
                        continue
                    mean_auc += auroc(torch.sigmoid(pred_scores)[i], multi_hot_labels.int()[i], task = 'binary').item()

                if n_iters % self.checkpoint_every == 0:
                    end = time.time()
                    speed = (n_samples_processed - n_samples_previously_processed) / (end - start)
                    start = time.time()
                    n_samples_previously_processed = n_samples_processed
                    print(f"{prefix} loss: {mean_loss/n_samples_processed:0.3f}, auc: {mean_auc/n_samples_processed:0.3f}, speed: {speed} samples/s")

                    if optimizer:
                        print("Checkpointing model")
                        torch.save(policy, self.checkpoint)

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if n_samples_processed > 0:
            mean_loss /= n_samples_processed
            mean_acc /= n_samples_processed
            mean_auc /= n_samples_processed
            mean_offby /= n_samples_processed
            for i in range(len(k_acc)):
                k_acc[i] /= n_samples_processed
        else:
            mean_loss = float("inf")
            mean_acc = 0
            mean_offby = float("inf")
            mean_auc = 0
            for i in range(len(k_acc)):
                k_acc[i] = 0

        print("n_samples_processed", n_samples_processed)
        return mean_loss, mean_auc    

    def train(self, num_epochs=30, lr=0.001, warmstart=None, 
                    detect_anomalies=False, anneal_lr=False, 
                    give_up_after=100, decay_lr_after=20):

        print(F"Using DEVICE {self.device}")
        tb_writer = SummaryWriter(log_dir=self.tensorboard, comment="neural_LNS")
        policy = GNNPolicy(self.gnn_type).to(self.device)

        if not (warmstart is None):
            print("Warnstarting training, loading from checkpoint %s"%(warmstart))
            policy = self.load_policy_from_checkpoint(warmstart)
            policy = policy.to(self.device)

        print(f"Checkpoint will be saved to {self.checkpoint}")

        num_of_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("number of parameters =", num_of_parameters)

        learning_rate = lr
        best_valid_loss = float("inf")
        last_improved = 0
        optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.00005, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate/10, verbose=False)

        for epoch in range(num_epochs):
            start = time.time()
            print(f"Starting epoch {epoch+1}", flush=True)

            with torch.autograd.set_detect_anomaly(detect_anomalies):
                train_iterator = self.train_loader.load(batch_size=self.batch_size)
                train_loss, train_auc = self.process(policy, train_iterator, optimizer)
            print(f"Train loss: {train_loss:0.3f}, Train auc: {train_auc:0.3f}")

            valid_iterator = self.valid_loader.load(batch_size=self.batch_size)
            valid_loss, valid_auc = self.process(policy, valid_iterator, None)
            print(f"Valid loss: {valid_loss:0.3f}, Valid auc: {valid_auc:0.3f}")

            end = time.time()

            tb_writer.add_scalar("Train/Loss", train_loss, global_step=epoch)
            tb_writer.add_scalar("Train/Auc", train_auc, global_step=epoch)
            tb_writer.add_scalar("Valid/Loss", valid_loss, global_step=epoch)
            tb_writer.add_scalar("Valid/Auc", valid_auc, global_step=epoch)

            # Done with one epoch, we can freeze the normalization
            policy.freeze_normalization()
            # Anneal the learning rate if requested
            if anneal_lr:
                scheduler.step()

            # Save the trained model
            print(f"Done with epoch {epoch+1} in {end-start:.1f}s, checkpointing model", flush=True)
            torch.save(policy, self.checkpoint+"_epoch%d"%(epoch))

            # Check if we need to abort, adjust the learning rate, or just give up
            if math.isnan(train_loss) or math.isnan(valid_loss):
                print("NaN detected in loss, aborting")
                break
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                last_improved = epoch
                print("Checkpointing new best model in " + self.checkpoint + "_best")
                torch.save(policy, self.checkpoint + "_best")
            elif epoch - last_improved > give_up_after:
                print("Validation loss didn't improve for too many epochs, giving up")
                break
            elif epoch - last_improved > decay_lr_after:
                learning_rate /= 2
                print(f"Adjusting the learning rate to {learning_rate}")
                optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate/10, verbose=False)
                # Give the model some time to improve with the new learning rate
                last_improved = epoch