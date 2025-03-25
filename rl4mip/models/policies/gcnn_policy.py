import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BipartiteGCNN(nn.Module):
    """Bipartite Graph Convolutional Network for branching.
    
    Based on the paper: "Exact Combinatorial Optimization with Graph Convolutional Neural Networks"
    by Gasse et al. (NeurIPS 2019)
    """
    
    def __init__(self, col_dim=19, row_dim=14, edge_dim=1, hidden_dim=64, device=None):
        """Initialize the GCNN model.
        
        Args:
            col_dim: Dimension of column features (variables)
            row_dim: Dimension of row features (constraints)
            edge_dim: Dimension of edge features (coefficients)
            hidden_dim: Hidden dimension for the network
            device: Device to use (e.g., 'cpu', 'cuda:0')
        """
        super(BipartiteGCNN, self).__init__()
        
        # Set device
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Store dimensions
        self.col_dim = col_dim
        self.row_dim = row_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        print(f"Initializing GCNN with dimensions: col={col_dim}, row={row_dim}, edge={edge_dim}")
        
        # Initial embeddings
        self.col_embedding = nn.Sequential(
            nn.Linear(col_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.row_embedding = nn.Sequential(
            nn.Linear(row_dim, hidden_dim),
            nn.ReLU()
        )
        
        # V → C convolution (variables to constraints)
        self.v2c_conv = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # C → V convolution (constraints to variables)
        self.c2v_conv = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, 1)
        
        # Move model to the specified device
        self.to(self.device)
    
    def forward(self, state):
        """Forward pass through the network.
        
        Args:
            state: Dictionary containing:
                - col_features: Variable features tensor [n_cols, col_dim]
                - row_features: Constraint features tensor [n_rows, row_dim]
                - edge_features: Edge features tensor [n_edges, edge_dim]
                - edge_indices: Edge indices tensor [2, n_edges]
                - branchable_mask: Boolean mask of branchable variables [n_cols]
                
        Returns:
            Tensor of scores for each variable [n_cols]
        """
        # Extract features
        col_features = state['col_features']  # (num_cols, col_dim)
        row_features = state['row_features']  # (num_rows, row_dim)
        edge_features = state['edge_features']  # (num_edges, edge_dim)
        edge_indices = state['edge_indices']  # (2, num_edges)
        
        # Initial embeddings
        col_embeds = self.col_embedding(col_features)  # (num_cols, hidden_dim)
        row_embeds = self.row_embedding(row_features)  # (num_rows, hidden_dim)
        
        # Check if there are edges to process
        if edge_indices.shape[1] > 0:
            # Extract indices
            row_indices, col_indices = edge_indices
            
            # Variables to Constraints convolution
            v2c_features = torch.cat([
                col_embeds[col_indices],
                edge_features,
                row_embeds[row_indices]
            ], dim=1)
            v2c_messages = self.v2c_conv(v2c_features)
            
            # Aggregate v2c messages at constraint nodes
            new_row_embeds = row_embeds.clone()
            for i in range(len(row_indices)):
                row_idx = row_indices[i]
                new_row_embeds[row_idx] += v2c_messages[i]
            
            # Constraints to Variables convolution
            c2v_features = torch.cat([
                new_row_embeds[row_indices],
                edge_features,
                col_embeds[col_indices]
            ], dim=1)
            c2v_messages = self.c2v_conv(c2v_features)
            
            # Aggregate c2v messages at variable nodes
            new_col_embeds = col_embeds.clone()
            for i in range(len(col_indices)):
                col_idx = col_indices[i]
                new_col_embeds[col_idx] += c2v_messages[i]
        else:
            # If no edges, just use the initial embeddings
            new_col_embeds = col_embeds
        
        # Final prediction
        scores = self.output(new_col_embeds).squeeze(-1)
        
        return scores



class GCNNBranchingPolicy:
    """Graph Convolutional Neural Network policy for variable branching.
    
    Can be used as a drop-in replacement for the RandomBranchingPolicy.
    """
    
    def __init__(self, pretrained_path=None, hidden_dim=64, device=None):
        """Initialize the GCNN branching policy.
        
        Args:
            pretrained_path: Path to a pretrained model (optional)
            hidden_dim: Hidden dimension for the network
            device: Device to use (e.g., 'cpu', 'cuda:0') - auto-detected if not specified
        """
        # Set device
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Log device info
        if self.device.startswith('cuda'):
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for GCNN policy")
        
        # Initialize model
        self.model = BipartiteGCNN(hidden_dim=hidden_dim, device=self.device)
        
        # Load pretrained model if provided
        if pretrained_path:
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        
        # Set to evaluation mode
        self.model.eval()
    
    def act(self, state):
        """Select a variable to branch on.
        
        Args:
            state: State from SCIP's getBipartiteGraphRepresentation
            
        Returns:
            Index of the selected variable or None
        """
        # Check if state is valid
        if state is None:
            return None
        
        # Check if there are branchable variables
        if sum(state['branchable_mask']) == 0:
            return None
        
        try:
            # Process the state
            proc_state = self._process_state(state)
            
            # Get scores from the model
            with torch.no_grad():
                scores = self.model(proc_state)
            
            # Apply mask and select the best variable
            branchable_mask = proc_state['branchable_mask']
            scores[~branchable_mask] = float('-inf')
            
            # Get branchable indices and select the highest scoring one
            branchable_indices = torch.nonzero(branchable_mask).squeeze(-1)
            if len(branchable_indices) == 0:
                return None
                
            # Select highest scoring variable among branchable ones
            branchable_scores = scores[branchable_indices]
            best_idx = branchable_indices[torch.argmax(branchable_scores)].item()
            
            return best_idx
            
        except Exception as e:
            print(f"Error in GCNN policy: {e}")
            return None
    
    def _process_state(self, state):
        """Process the state from SCIP to tensor format.
        
        Args:
            state: State from SCIP's getBipartiteGraphRepresentation
            
        Returns:
            Processed state with tensors
        """
        # Convert features to tensors and move to device
        col_features = torch.FloatTensor(state['col_features']).to(self.device)
        row_features = torch.FloatTensor(state['row_features']).to(self.device)
        
        # Process edge features and create edge indices
        edge_indices = []
        edge_features_list = []
        
        for edge in state['edge_features']:
            col_idx = int(edge[0])  # col_idx
            row_idx = int(edge[1])  # row_idx
            coef = float(edge[2])   # coefficient
            
            edge_indices.append([row_idx, col_idx])
            edge_features_list.append([coef])
        
        # Handle edge case of empty graph
        if not edge_indices:
            edge_indices = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            edge_features = torch.zeros((0, 1), dtype=torch.float32).to(self.device)
        else:
            edge_indices = torch.LongTensor(edge_indices).t().to(self.device)  # Shape: (2, num_edges)
            edge_features = torch.FloatTensor(edge_features_list).to(self.device)  # Shape: (num_edges, 1)
        
        # Create branchable mask tensor
        branchable_mask = torch.BoolTensor(state['branchable_mask']).to(self.device)
        
        # Return processed state
        return {
            'col_features': col_features,
            'row_features': row_features,
            'edge_features': edge_features,
            'edge_indices': edge_indices,
            'branchable_mask': branchable_mask
        }