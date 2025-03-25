"""
Train GCNN branching policy using imitation learning from collected strong branching data.
"""

import os
import sys
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl4mip.models.policies.gcnn_policy import BipartiteGCNN


class BranchingDataset(Dataset):
    """Dataset for imitation learning from strong branching."""
    
    def __init__(self, states, actions):
        """Initialize the dataset.
        
        Args:
            states: List of state dictionaries
            actions: List of expert actions (column indices)
        """
        self.states = states
        self.actions = actions
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.actions)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (state, action) pair
        """
        return self.states[idx], self.actions[idx]


def process_state(state, device):
    """Process a state dictionary into tensors.
    
    Args:
        state: Dictionary with state information
        device: Device to put tensors on
        
    Returns:
        dict: Processed state with tensors
    """
    # Process column features
    col_features = torch.FloatTensor(state["col_features"]).to(device)
    
    # Process row features
    row_features = torch.FloatTensor(state["row_features"]).to(device)
    
    # Process edge features and indices
    edge_indices = []
    edge_features_list = []
    
    for edge in state["edge_features"]:
        col_idx = int(edge[0])  # col_idx
        row_idx = int(edge[1])  # row_idx
        coef = float(edge[2])   # coefficient
        
        edge_indices.append([row_idx, col_idx])
        edge_features_list.append([coef])
    
    # Handle edge case of empty graph
    if not edge_indices:
        edge_indices = torch.zeros((2, 0), dtype=torch.long).to(device)
        edge_features = torch.zeros((0, 1), dtype=torch.float32).to(device)
    else:
        edge_indices = torch.LongTensor(edge_indices).t().to(device)  # Shape: (2, num_edges)
        edge_features = torch.FloatTensor(edge_features_list).to(device)  # Shape: (num_edges, 1)
    
    # Create branchable mask tensor
    branchable_mask = torch.BoolTensor(state["branchable_mask"]).to(device)
    
    return {
        "col_features": col_features,
        "row_features": row_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "branchable_mask": branchable_mask
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train the model for one epoch.
    
    Args:
        model: GCNN model
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on
        
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (states, targets) in enumerate(dataloader):
        # Process each state in the batch individually
        batch_loss = 0
        batch_correct = 0
        batch_total = 0
        
        optimizer.zero_grad()
        
        for state, target in zip(states, targets):
            processed_state = process_state(state, device)
            
            # Forward pass
            scores = model(processed_state)
            
            # Apply mask (consider only branchable variables)
            mask = processed_state["branchable_mask"]
            
            # Skip if no branchable variables (shouldn't happen in collected data)
            if not mask.any():
                continue
                
            valid_indices = torch.where(mask)[0]
            valid_scores = scores[valid_indices]
            
            # Find where the target is in valid_indices
            target_idx = -1
            for i, idx in enumerate(valid_indices):
                if idx.item() == target.item():
                    target_idx = i
                    break
                    
            # Skip if target not in valid indices (shouldn't happen with collected data)
            if target_idx == -1:
                continue
                
            # Create target tensor - giving the index within valid_indices
            target_tensor = torch.tensor([target_idx], device=device)
            
            # Calculate loss on valid scores only
            loss = criterion(valid_scores.unsqueeze(0), target_tensor)
            loss.backward()
            
            batch_loss += loss.item()
            
            # Check accuracy
            pred_idx = torch.argmax(valid_scores).item()
            pred_col_idx = valid_indices[pred_idx].item()
            batch_correct += (pred_col_idx == target.item())
            batch_total += 1
        
        # Update model weights
        optimizer.step()
        
        # Update statistics
        total_loss += batch_loss
        correct += batch_correct
        total += batch_total
        
        # Print progress
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {batch_loss/max(1, batch_total):.4f} | "
                  f"Accuracy: {100*batch_correct/max(1, batch_total):.2f}%")
    
    return total_loss / max(1, total), correct / max(1, total)


def validate(model, dataloader, criterion, device):
    """Validate the model.
    
    Args:
        model: GCNN model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for states, targets in dataloader:
            # Process each state in the batch individually
            for state, target in zip(states, targets):
                processed_state = process_state(state, device)
                
                # Forward pass
                scores = model(processed_state)
                
                # Apply mask (consider only branchable variables)
                mask = processed_state["branchable_mask"]
                
                # Skip if no branchable variables
                if not mask.any():
                    continue
                    
                valid_indices = torch.where(mask)[0]
                valid_scores = scores[valid_indices]
                
                # Find where the target is in valid_indices
                target_idx = -1
                for i, idx in enumerate(valid_indices):
                    if idx.item() == target.item():
                        target_idx = i
                        break
                        
                # Skip if target not in valid indices (shouldn't happen with collected data)
                if target_idx == -1:
                    continue
                    
                # Create target tensor - giving the index within valid_indices
                target_tensor = torch.tensor([target_idx], device=device)
                
                # Calculate loss on valid scores only
                loss = criterion(valid_scores.unsqueeze(0), target_tensor)
                
                total_loss += loss.item()
                
                # Check accuracy
                pred_idx = torch.argmax(valid_scores).item()
                pred_col_idx = valid_indices[pred_idx].item()
                correct += (pred_col_idx == target.item())
                total += 1
    
    return total_loss / max(1, total), correct / max(1, total)


def collate_fn(batch):
    """Custom collate function for the dataloader.
    
    Args:
        batch: Batch of (state, action) pairs
        
    Returns:
        tuple: (states, actions)
    """
    states, actions = zip(*batch)
    actions = torch.tensor(actions, dtype=torch.long)
    return states, actions


def main():
    """Train the GCNN model using imitation learning."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load collected data
    data_path = os.path.join(os.path.dirname(__file__), "data/strong_branching_data.pkl")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    states = data["states"]
    actions = data["actions"]
    
    print(f"Loaded {len(states)} samples")
    
    # Inspect the first state to determine feature dimensions
    if len(states) > 0:
        first_state = states[0]
        col_dim = len(first_state["col_features"][0])  # Number of features per column
        row_dim = len(first_state["row_features"][0])  # Number of features per row
        edge_dim = 1  # We use only the coefficient as edge feature
        
        print(f"Feature dimensions: col={col_dim}, row={row_dim}, edge={edge_dim}")
        
        # Print more details about the first state for debugging
        print(f"First state details:")
        print(f"  Number of columns: {len(first_state['col_features'])}")
        print(f"  Number of rows: {len(first_state['row_features'])}")
        print(f"  Number of edges: {len(first_state['edge_features'])}")
        print(f"  Number of branchable variables: {sum(first_state['branchable_mask'])}")
    else:
        print("Error: No data samples found")
        return
    
    # Create full dataset
    full_dataset = BranchingDataset(states, actions)
    
    # Split into training and validation sets (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = BipartiteGCNN(
        col_dim=col_dim,
        row_dim=row_dim,
        edge_dim=edge_dim,
        hidden_dim=64,
        device=device
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 20
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStarting training...")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_time = time.time() - start_time
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {100*train_acc:.2f}% | Time: {train_time:.2f}s")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {100*val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            model_path = os.path.join(output_dir, "gcnn_imitation.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with validation accuracy: {100*val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs")
                break
    
    print(f"\nTraining completed! Best validation accuracy: {100*best_val_acc:.2f}%")


if __name__ == "__main__":
    main()