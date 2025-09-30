import os
import sys
import torch
import torch_geometric
from pathlib import Path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ml4co.Trainer.Nodeselect_model.ML.model import GNNPolicy, RankNet
from DataLoader import GNNDataLoader, RankNetDataLoader, SVMDataLoader
import numpy as np
import sklearn as sk
from sklearn import svm, datasets
from joblib import dump, load
from torch.utils.data import DataLoader, TensorDataset



class ML_Nodeselect_Trainer:
    def __init__(self, problem='GISP', datapath=None, model_dir=None, device='cpu'):
        self.problem = problem
        self.datapath = datapath
        self.model_dir = model_dir
        self.device = device
    
    def train(self, method='gnn', lr=0.005, n_epoch=2, patience=10, early_stopping=20, normalize=True, 
                batch_train = 16, batch_valid  = 256, loss_fn = torch.nn.BCELoss(), optimizer_fn = torch.optim.Adam):
        if method == 'gnn':
            return self.train_gnn(problem=self.problem, lr=lr, n_epoch=n_epoch, patience=patience, early_stopping=early_stopping, 
                                    normalize=normalize, device=self.device, datapath=self.datapath, model_dir=self.model_dir, 
                                    batch_train = batch_train, batch_valid  = batch_valid, loss_fn = loss_fn, optimizer_fn = optimizer_fn)
        elif method == 'ranknet':
            return self.train_ranknet(problem=self.problem, lr=lr, n_epoch=n_epoch, patience=patience, early_stopping=early_stopping, 
                                        normalize=normalize, device=self.device, datapath=self.datapath, model_dir=self.model_dir, 
                                        batch_train = batch_train, batch_valid  = batch_valid, loss_fn = loss_fn, optimizer_fn = optimizer_fn)
        elif method == 'svm':
            return self.train_svm(problem=self.problem, datapath=self.datapath, model_dir=self.model_dir)


    def train_gnn(self, problem, lr, n_epoch, patience, early_stopping, normalize, device, datapath, 
                    model_dir,batch_train, batch_valid, loss_fn, optimizer_fn):
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        train_files = [ str(path) for path in Path(os.path.join(datapath, 
                                                                f"behaviours/{problem}/train")).glob("*.pt") ]
        
        valid_files = [ str(path) for path in Path(os.path.join(datapath, 
                                                                f"behaviours/{problem}/valid")).glob("*.pt") ]


        # train_data = GraphDataset(train_files)
        # valid_data = GraphDataset(valid_files)

        # train_loader = torch_geometric.loader.DataLoader(train_data, 
        #                                                 batch_size=batch_train, 
        #                                                 shuffle=True, 
        #                                                 follow_batch=['constraint_features_s', 
        #                                                             'constraint_features_t',
        #                                                             'variable_features_s',
        #                                                             'variable_features_t'])
        
        # valid_loader = torch_geometric.loader.DataLoader(valid_data, 
        #                                                 batch_size=batch_valid, 
        #                                                 shuffle=False, 
        #                                                 follow_batch=['constraint_features_s',
        #                                                             'constraint_features_t',
        #                                                             'variable_features_s',
        #                                                             'variable_features_t'])
        train_loader = GNNDataLoader( train_files,
                                        batch_size=batch_train, 
                                        shuffle=True, 
                                        follow_batch=['constraint_features_s',
                                                    'constraint_features_t',
                                                    'variable_features_s',
                                                    'variable_features_t'])
        valid_loader = GNNDataLoader( valid_files,
                                        batch_size=batch_valid, 
                                        shuffle=False, 
                                        follow_batch=['constraint_features_s',
                                                    'constraint_features_t',
                                                    'variable_features_s',
                                                    'variable_features_t'])


        policy = GNNPolicy().to(device)

        optimizer = optimizer_fn(policy.parameters(), lr=lr) #ADAM is the best

        print("-------------------------")
        print(f"GNN for problem {problem}")
        print(f"Training on:          {len(train_files)} samples")
        print(f"Validating on:        {len(valid_files)} samples")
        print(f"Batch Size Train:     {batch_train}")
        print(f"Batch Size Valid      {batch_valid}")
        print(f"Learning rate:        {lr} ")
        print(f"Number of epochs:     {n_epoch}")
        print(f"Normalize:            {normalize}")
        print(f"Device:               {device}")
        print(f"Loss fct:             {loss_fn}")
        print(f"Optimizer:            {optimizer_fn}")  
        print(f"Model's Size:         {sum(p.numel() for p in policy.parameters())} parameters ")
        print("-------------------------") 

        # 训练
        for epoch in range(n_epoch):
            print(f"Epoch {epoch + 1}")
            
            train_loss, train_acc = self.process(policy, 
                                            train_loader, 
                                            loss_fn,
                                            device,
                                            optimizer=optimizer, 
                                            normalize=normalize)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )
        
            valid_loss, valid_acc = self.process(policy, 
                                            valid_loader, 
                                            loss_fn, 
                                            device,
                                            optimizer=None,
                                            normalize=normalize)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            
            print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )
        
        import datetime 
        time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        model_path_directory = Path(os.path.join(model_dir, f'node_selection'))
        model_path_directory.mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_dir, f'node_selection/policy_{problem}_gnn.pkl')

        torch.save(policy.state_dict(),model_path)

        return model_path


    def train_ranknet(self, problem, lr, n_epoch, patience, early_stopping, normalize, device, datapath, model_dir, 
                        batch_train, batch_valid, loss_fn, optimizer_fn):

        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        train_files = [ str(path) for path in Path(os.path.join(datapath, 
                                                                f"behaviours_svm/{problem}/train")).glob("*.csv") ]
        
        valid_files = [ str(path) for path in Path(os.path.join(datapath, 
                                                                f"behaviours_svm/{problem}/valid")).glob("*.csv") ]


        # X_train, y_train, _ = self.get_data(train_files)
        # X_valid, y_valid, _ = self.get_data(valid_files)
        
        # X_train = torch.from_numpy(X_train)
        # y_train = torch.from_numpy(y_train).unsqueeze(1)

        # X_valid = torch.from_numpy(X_valid)
        # y_valid = torch.from_numpy(y_valid).unsqueeze(1)
          
        train_data_loader = RankNetDataLoader(train_files)
        X_train, y_train = train_data_loader.dataloader()

        valid_data_loader = RankNetDataLoader(valid_files)
        X_valid, y_valid = valid_data_loader.dataloader()



        policy = RankNet().to(device)
        optimizer = optimizer_fn(policy.parameters(), lr=lr) #ADAM is the best

        print("-------------------------")
        print(f"Ranknet for problem {problem}")
        print(f"Training on:          {len(X_train)} samples")
        print(f"Validating on:        {len(X_valid)} samples")
        print(f"Batch Size Train:     {1}")
        print(f"Batch Size Valid      {1}")
        print(f"Learning rate:        {lr} ")
        print(f"Number of epochs:     {n_epoch}")
        print(f"Normalize:            {normalize}")
        print(f"Device:               {device}")
        print(f"Loss fct:             {loss_fn}")
        print(f"Optimizer:            {optimizer_fn}")  
        print(f"Model's Size:         {sum(p.numel() for p in policy.parameters())} parameters ")
        print("-------------------------") 

        for epoch in range(n_epoch):
            print(f"Epoch {epoch + 1}")
            
            train_loss, train_acc = self.process_ranknet(policy, 
                                            X_train, y_train, 
                                            loss_fn,
                                            device,
                                            optimizer=optimizer)

            # def process_ranknet_v1(self, policy, X, y, loss_fct, device, batch_size=16, optimizer=None):
            # train_loss, train_acc = self.process_ranknet_v1(policy, 
            #                                 X_train, y_train, 
            #                                 loss_fn,
            #                                 device,
            #                                 batch_size = batch_train,
            #                                 optimizer=optimizer)
            

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )
        
            valid_loss, valid_acc = self.process_ranknet(policy, 
                                            X_valid, y_valid, 
                                            loss_fn, 
                                            device,
                                            optimizer=None)
            # valid_loss, valid_acc = self.process_ranknet_v1(policy, 
            #                                 X_valid, y_valid, 
            #                                 loss_fn, 
            #                                 device,
            #                                 batch_size = batch_valid,
            #                                 optimizer=None)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            
            print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )
        
        import datetime 
        time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        model_path_directory = Path(os.path.join(model_dir, f'node_selection'))
        model_path_directory.mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_dir, f'node_selection/policy_{problem}_ranknet.pkl')

        torch.save(policy.state_dict(), model_path)

        return model_path

    def train_svm(self, problem, datapath, model_dir):
        train_files = [ str(path) for path in Path(os.path.join(datapath, 
                                                                f"behaviours_svm/{problem}/train")).glob("*.csv") ]
        
        valid_files = [ str(path) for path in Path(os.path.join(datapath, 
                                                                f"behaviours_svm/{problem}/valid")).glob("*.csv") ]

        # X, y, depths = self.get_data(train_files)
        # X_valid, y_valid, depths_valid = self.get_data(valid_files)

        # print("len(train_files) = ", len(train_files))
        train_data_loader = SVMDataLoader(train_files)
        X, y, depths = train_data_loader.dataloader()

        valid_data_loader = SVMDataLoader(valid_files)
        X_valid, y_valid, depths_valid = valid_data_loader.dataloader()


        model = svm.LinearSVC()

        # print("X.shape = ", X.shape)
        
        model.fit(X,y, np.exp(2.67/np.min(depths, axis=1)))


        try:
            valid_acc = model.score(X_valid,y_valid, np.min(depths_valid, axis=1))
        except :
            valid_acc = model.score(X,y, np.min(depths, axis=1))
            
        
        print(f"Accuracy on validation set : {valid_acc}")
        
        import datetime 
        time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        model_path_directory = Path(os.path.join(model_dir, f'node_selection'))
        model_path_directory.mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_dir, f'node_selection/policy_{problem}_svm.pkl')

        dump(model, model_path)

        return model_path



    def normalize_graph(self, 
                        constraint_features, 
                        edge_index,
                        edge_attr,
                        variable_features,
                        bounds,
                        depth,
                        bound_normalizor = 1000):
        
        
        #SMART
        obj_norm = torch.max(torch.abs(variable_features[:,2]), axis=0)[0].item()
        var_max_bounds = torch.max(torch.abs(variable_features[:,:2]), axis=1, keepdim=True)[0]  
        
        var_max_bounds.add_(var_max_bounds == 0)
        
        var_normalizor = var_max_bounds[edge_index[0]]
        cons_normalizor = constraint_features[edge_index[1], 0:1]
        normalizor = var_normalizor/(cons_normalizor + (cons_normalizor == 0))
        
        variable_features[:,2].div_(obj_norm)
        variable_features[:,:2].div_(var_max_bounds)
        constraint_features[:,0].div_(constraint_features[:,0] + (constraint_features[:,0] == 0) )
        edge_attr.mul_(normalizor)
        bounds.div_(bound_normalizor)
        
        return (constraint_features, edge_index, edge_attr, variable_features, bounds, depth)



    #function definition
    # https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
    def process(self, policy, data_loader, loss_fct, device, optimizer=None, normalize=True):
        """
        This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
        """
        mean_loss = 0
        mean_acc = 0
        n_samples_processed = 0
        with torch.set_grad_enabled(optimizer is not None):
            for idx,batch in enumerate(data_loader):
                batch = batch.to(device)
                if normalize:
                    #IN place operations
                    (batch.constraint_features_s,
                    batch.edge_index_s, 
                    batch.edge_attr_s,
                    batch.variable_features_s,
                    batch.bounds_s,
                    batch.depth_s)  =  self.normalize_graph(batch.constraint_features_s,  batch.edge_index_s, batch.edge_attr_s,
                                                        batch.variable_features_s, batch.bounds_s,  batch.depth_s)
                    
                    (batch.constraint_features_t,
                    batch.edge_index_t, 
                    batch.edge_attr_t,
                    batch.variable_features_t,
                    batch.bounds_t,
                    batch.depth_t)  =  self.normalize_graph(batch.constraint_features_t,  batch.edge_index_t, batch.edge_attr_t,
                                                        batch.variable_features_t, batch.bounds_t,  batch.depth_t)
                                                        
            
                y_true = 0.5*batch.y + 0.5 #0,1 label from -1,1 label
                y_proba = policy(batch)
                y_pred = torch.round(y_proba)
                
                # Compute the usual cross-entropy classification loss
                #loss_fct.weight = torch.exp((1+torch.abs(batch.depth_s - batch.depth_t)) / 
                                #(torch.min(torch.vstack((batch.depth_s,  batch.depth_t)), axis=0)[0]))

                l = loss_fct(y_proba, y_true)
                loss_value = l.item()
                if optimizer is not None:
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                
                y_proba[y_proba >= 0.5] = 1.0
                y_proba[y_proba < 0.5] = 0.0
                accuracy = (y_proba == y_true).float().mean()

                mean_loss += loss_value * batch.num_graphs
                mean_acc += accuracy * batch.num_graphs
                n_samples_processed += batch.num_graphs
                #print(y_proba.item(), y_true.item())

        mean_loss /= (n_samples_processed + ( n_samples_processed == 0))
        mean_acc /= (n_samples_processed  + ( n_samples_processed == 0))
        return mean_loss, mean_acc

    def process_ranknet(self, policy, X, y, loss_fct, device, optimizer=None):
        """
        This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
        """
        mean_loss = 0
        mean_acc = 0
        n_samples_processed = 0
        X.to(device)


        with torch.set_grad_enabled(optimizer is not None):
            for idx,x in enumerate(X):
                yi = y[idx].to(device)
                y_true = 0.5*yi + 0.5 #0,1 label from -1,1 label
                # print("X:", X)          

                y_proba = policy(x[:20].to(device), x[20:].to(device), device)
                y_pred = torch.round(y_proba)
                
                # Compute the usual cross-entropy classification loss
                #loss_fct.weight = torch.exp((1+torch.abs(batch.depth_s - batch.depth_t)) / 
                                #(torch.min(torch.vstack((batch.depth_s,  batch.depth_t)), axis=0)[0]))
                # print(y_proba.dtype)
                # 修改代码
                y_proba = y_proba.to(torch.float32)
                y_true = y_true.to(torch.float32)
                # print("y_proba:", y_proba)
                # print("y_true:", y_true)
                l = loss_fct(y_proba, y_true)
                #print(l)
                loss_value = l.item()
                if optimizer is not None:
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                
                y_proba[y_proba >= 0.5] = 1.0
                y_proba[y_proba < 0.5] = 0.0
                accuracy = (y_proba == y_true).float().mean()

                mean_loss += loss_value
                mean_acc += accuracy 
                n_samples_processed += 1
                #print(y_proba.item(), y_true.item())

        mean_loss /= (n_samples_processed + ( n_samples_processed == 0))
        mean_acc /= (n_samples_processed  + ( n_samples_processed == 0))
        return mean_loss, mean_acc

    def process_ranknet_v1(self, policy, X, y, loss_fct, device, batch_size=16, optimizer=None):
        """
        使用PyTorch DataLoader实现batch处理的版本 (推荐)
        
        Args:
            policy: 模型
            X: 输入数据 tensor
            y: 标签 tensor  
            loss_fct: 损失函数
            device: 设备
            batch_size: 批次大小，默认32
            optimizer: 优化器，如果为None则为验证模式
        """
        mean_loss = 0
        mean_acc = 0
        n_samples_processed = 0
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(optimizer is not None),  # 训练时打乱，验证时不打乱
            drop_last=False  # 保留最后一个不完整的batch
        )
        
        with torch.set_grad_enabled(optimizer is not None):
            for batch_x, batch_y in dataloader:
                # 将数据移到指定设备
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # 转换标签：从 -1,1 到 0,1
                y_true = 0.5 * batch_y + 0.5
                
                # 批量前向传播
                y_proba = policy(
                    batch_x[:, :20],  # 前20个特征
                    batch_x[:, 20:],  # 后20个特征  
                    device
                )
                
                # 确保数据类型一致
                y_proba = y_proba.to(torch.float32)
                y_true = y_true.to(torch.float32)
                
                # 计算损失
                loss = loss_fct(y_proba, y_true)
                loss_value = loss.item()
                
                # 反向传播和参数更新
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # 计算预测结果和准确率
                y_pred = (y_proba >= 0.5).float()
                accuracy = (y_pred == y_true).float().mean()
                
                # 累积统计
                current_batch_size = batch_x.size(0)
                mean_loss += loss_value * current_batch_size
                mean_acc += accuracy.item() * current_batch_size
                n_samples_processed += current_batch_size
        
        # 计算平均值
        mean_loss /= max(n_samples_processed, 1)
        mean_acc /= max(n_samples_processed, 1)
        
        return mean_loss, mean_acc