import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

# ========== Nature CNN ==========
class nature_cnn(nn.Module):
    def __init__(self, input_channels=4, **conv_kwargs):
        super(nature_cnn, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, **conv_kwargs)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, **conv_kwargs)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, **conv_kwargs)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        
        # Initialize weights following TensorFlow's sqrt(2) scale
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1]:
            nn.init.orthogonal_(layer.weight, gain=torch.sqrt(torch.tensor(2.0)))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.float() / 255.0  # Normalize input
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return x

# register("nature_cnn")(nature_cnn)

# ========== IMPALA CNN ==========
class ResidualBlock(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        return out + x

class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x

class build_impala_cnn(nn.Module):
    def __init__(self, input_channels=3, depths=[16, 32, 32], **conv_kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = input_channels
        for depth in depths:
            self.layers.append(ConvSequence(in_channels, depth))
            in_channels = depth
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * 8 * 8, 256)  # 这里假设输入是 64x64 图片

    def forward(self, x):
        x = x.float() / 255.0
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = F.relu(x)
        x = F.relu(self.fc(x))
        return x

# ========== MLP ==========
@register("mlp")
def mlp(num_layers=1, num_hidden=128, activation=torch.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)
    num_hidden: int                 size of fully-connected layers (default: 64)
    activation:                     activation function (default: torch.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor

    """
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.layers = nn.ModuleList()
            self.layer_norm = layer_norm
            self.activation = activation
            self.num_layers = num_layers
            self.num_hidden = num_hidden

            self.initialized = False  # 仅在 forward 时初始化第一层

        def forward(self, X):
            h = torch.flatten(X, start_dim=1)  # Flatten input
            
            # 仅在第一次 forward 时动态创建第一层
            if not self.initialized:
                input_dim = h.shape[1]  # 通过输入数据的形状推断 input_dim
                self.layers.append(nn.Linear(input_dim, self.num_hidden))
                for _ in range(1, self.num_layers):
                    self.layers.append(nn.Linear(self.num_hidden, self.num_hidden))
                    if self.layer_norm:
                        self.layers.append(nn.LayerNorm(self.num_hidden))
                self.initialized = True  # 标记已初始化

            for layer in self.layers:
                h = layer(h)
                if isinstance(layer, nn.Linear):  # 线性层后再加激活函数
                    h = self.activation(h)

            return h

    return MLP()

@register("gnn")
def gnn_policy():
    """
    Graph Neural Network policy module for structured data such as ILP problems.

    Returns:
    --------
    GNNPolicy class instance
    """
    class GNNPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            emb_size = 64
            cons_nfeats = 14
            edge_nfeats = 1
            var_nfeats = 22

            # CONSTRAINT EMBEDDING
            self.cons_embedding = nn.Sequential(
                nn.LayerNorm(cons_nfeats),
                nn.Linear(cons_nfeats, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
            )

            # EDGE EMBEDDING
            self.edge_embedding = nn.Sequential(
                nn.LayerNorm(edge_nfeats),
            )

            # VARIABLE EMBEDDING
            self.var_embedding = nn.Sequential(
                nn.LayerNorm(var_nfeats),
                nn.Linear(var_nfeats, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
            )

            # GNN layers
            self.conv_v_to_c = BipartiteGraphConvolution()
            self.conv_c_to_v = BipartiteGraphConvolution()
            self.conv_v_to_c2 = BipartiteGraphConvolution()
            self.conv_c_to_v2 = BipartiteGraphConvolution()

            # # Feature processing
            # self.output_module = nn.Sequential(
            #     nn.Linear(emb_size, emb_size),
            #     nn.ReLU(),
            #     nn.Linear(emb_size, emb_size),
            #     nn.ReLU(),
            #     nn.Linear(emb_size, emb_size)
            # )

            self.output_module = nn.Sequential(
                nn.Linear(emb_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

            self.global_pooling = nn.AdaptiveAvgPool1d(1)

            # Actor network outputs
            self.actor_mean = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, 1),
                nn.Tanh()
            )

            self.actor_logstd = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, 1),
                nn.Tanh()
            )

            self.norm_c = nn.LayerNorm(emb_size)
            
        def forward(self, constraint_features, edge_indices, edge_features, variable_features, batch_size):

            reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

            constraint_features = torch.tanh(self.norm_c(self.cons_embedding(constraint_features)))
            edge_features = self.edge_embedding(edge_features)
            variable_features = torch.tanh(self.norm_c(self.var_embedding(variable_features)))

            v1 = variable_features
            constraint_features = self.conv_v_to_c(
                v1, reversed_edge_indices, edge_features, constraint_features
            )

            v2 = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, v1
            )
            v2 = v2 + v1  # 🔁 残差

            constraint_features = self.conv_v_to_c2(
                v2, reversed_edge_indices, edge_features, constraint_features
            )

            v3 = self.conv_c_to_v2(
                constraint_features, edge_indices, edge_features, v2
            )
            v3 = v3 + v2  # 🔁 再加残差

            num_vars = v3.shape[0] // batch_size

            output = self.output_module(v3).view(batch_size, num_vars, -1)

            output = output * 0.6 + 0.2

            # constraint_features = self.conv_v_to_c(
            #     variable_features, reversed_edge_indices, edge_features, constraint_features
            # )
            # variable_features = self.conv_c_to_v(
            #     constraint_features, edge_indices, edge_features, variable_features
            # )
            # constraint_features = self.conv_v_to_c2(
            #     variable_features, reversed_edge_indices, edge_features, constraint_features
            # )
            # variable_features = self.conv_c_to_v2(
            #     constraint_features, edge_indices, edge_features, variable_features
            # )

            # num_vars = variable_features.shape[0] // batch_size

            # output = self.output_module(variable_features).view(batch_size, num_vars, -1)
            # output = output.permute(0, 2, 1)
            # output = self.global_pooling(output).squeeze(-1)

            return output

    return GNNPolicy()

@register("gnn_critic")
def gnn_policy_critic():
    """
    Graph Neural Network policy module for structured data such as ILP problems.

    Returns:
    --------
    GNNPolicy class instance
    """
    class GNNPolicy_critic(nn.Module):
        def __init__(self):
            super().__init__()
            emb_size = 64
            cons_nfeats = 14
            edge_nfeats = 1
            var_nfeats = 23

            # CONSTRAINT EMBEDDING
            self.cons_embedding = nn.Sequential(
                nn.LayerNorm(cons_nfeats),
                nn.Linear(cons_nfeats, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
            )

            # EDGE EMBEDDING
            self.edge_embedding = nn.Sequential(
                nn.LayerNorm(edge_nfeats),
            )

            # VARIABLE EMBEDDING
            self.var_embedding = nn.Sequential(
                nn.LayerNorm(var_nfeats),
                nn.Linear(var_nfeats, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
            )

            # GNN layers
            self.conv_v_to_c = BipartiteGraphConvolution()
            self.conv_c_to_v = BipartiteGraphConvolution()
            self.conv_v_to_c2 = BipartiteGraphConvolution()
            self.conv_c_to_v2 = BipartiteGraphConvolution()

            # # Feature processing
            # self.output_module = nn.Sequential(
            #     nn.Linear(emb_size, emb_size),
            #     nn.ReLU(),
            #     nn.Linear(emb_size, emb_size),
            #     nn.ReLU(),
            #     nn.Linear(emb_size, emb_size)
            # )

            self.output_module = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.Tanh(),
                nn.Linear(emb_size, 1),
                nn.Sigmoid()  # 最终在 0~1
            )

            self.global_pooling = nn.AdaptiveAvgPool1d(1)

            # Actor network outputs
            self.actor_mean = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, 1),
                nn.Tanh()
            )

            self.actor_logstd = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, 1),
                nn.Tanh()
            )

        def forward(self, constraint_features, edge_indices, edge_features, variable_features, batch_size):

            reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

            constraint_features = self.cons_embedding(constraint_features)
            edge_features = self.edge_embedding(edge_features.float())
            variable_features = self.var_embedding(variable_features)

            # constraint_features = self.conv_v_to_c(
            #     variable_features, reversed_edge_indices, edge_features, constraint_features
            # )
            # variable_features = self.conv_c_to_v(
            #     constraint_features, edge_indices, edge_features, variable_features
            # )
            # constraint_features = self.conv_v_to_c2(
            #     variable_features, reversed_edge_indices, edge_features, constraint_features
            # )
            # variable_features = self.conv_c_to_v2(
            #     constraint_features, edge_indices, edge_features, variable_features
            # )

            v1 = variable_features
            constraint_features = self.conv_v_to_c(
                v1, reversed_edge_indices, edge_features, constraint_features
            )

            v2 = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, v1
            )
            v2 = v2 + v1  # 🔁 残差

            constraint_features = self.conv_v_to_c2(
                v2, reversed_edge_indices, edge_features, constraint_features
            )

            v3 = self.conv_c_to_v2(
                constraint_features, edge_indices, edge_features, v2
            )
            v3 = v3 + v2  # 🔁 再加残差

            num_vars = v3.shape[0] // batch_size

            output = self.output_module(v3).view(batch_size, num_vars, -1)

            output = output.permute(0, 2, 1)
            output = self.global_pooling(output).squeeze(-1)

            return output

    return GNNPolicy_critic()

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output

def mlp(num_layers=1, num_hidden=128, activation=F.tanh, layer_norm=False):
    return mlp(num_layers, num_hidden, activation, layer_norm)

@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(X):
        return nature_cnn(X, **conv_kwargs)
    return network_fn

@register("impala_cnn")
def impala_cnn(**conv_kwargs):
    def network_fn(X):
        return build_impala_cnn(X, **conv_kwargs)
    return network_fn

class cnn_small(nn.Module):
    def __init__(self, **conv_kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.fc = nn.Linear(16 * 9 * 9, 128)  # 9x9 assumes input size 64x64

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        return F.relu(self.fc(x))

@register("cnn_small")
def cnn_small(**conv_kwargs):
    return cnn_small(**conv_kwargs)

class LSTMNet(nn.Module):
    def __init__(self, input_size, nlstm=128, layer_norm=False):
        super().__init__()
        self.nlstm = nlstm
        self.layer_norm = layer_norm
        self.lstm = nn.LSTM(input_size, nlstm, batch_first=True)
        self.initial_state = torch.zeros((1, 2, nlstm), dtype=torch.float32)

    def forward(self, x, state=None, mask=None):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten input
        
        if state is None:
            state = (torch.zeros(1, batch_size, self.nlstm),
                     torch.zeros(1, batch_size, self.nlstm))
        
        output, new_state = self.lstm(x.unsqueeze(1), state)
        return output.squeeze(1), {'state': new_state, 'initial_state': self.initial_state}

@register("lstm")
def lstm(nlstm=128, layer_norm=False):
    return LSTMNet(input_size=1, nlstm=nlstm, layer_norm=layer_norm)

@register("cnn_lstm")
def cnn_lstm(nlstm=128, layer_norm=False, conv_fn=None, **conv_kwargs):
    class CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = conv_fn(**conv_kwargs) if conv_fn else nn.Identity()
            self.lstm = LSTMNet(input_size=128, nlstm=nlstm, layer_norm=layer_norm)

        def forward(self, x, state=None, mask=None):
            x = self.conv(x)
            return self.lstm(x, state, mask)
    
    return CNNLSTM()

@register("impala_cnn_lstm")
def impala_cnn_lstm():
    return cnn_lstm(nlstm=256, conv_fn=build_impala_cnn)

@register("cnn_lnlstm")
def cnn_lnlstm(nlstm=128, **conv_kwargs):
    return cnn_lstm(nlstm, layer_norm=True, **conv_kwargs)

@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_channels = 3  # Assuming RGB input
            for out_channels, kernel_size, stride in convs:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
                layers.append(nn.ReLU())
                in_channels = out_channels
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            x = x.float() / 255.0
            return self.network(x)
    
    return ConvNet()

def get_network_builder(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError(f'Unknown network type: {name}')