import torch
import numpy as np

class RandomBranchingPolicy:
    """Simple random policy for selecting branching variables"""
    
    def act(self, state):
        """Select a random branchable LP column"""
        if not state:
            return None
        
        # Get branchable mask
        branchable_mask = state['branchable_mask']
        
        # Find valid indices
        valid_indices = torch.where(branchable_mask)[0]
        
        if len(valid_indices) == 0:
            return None
        
        # Select random index
        idx = np.random.randint(0, len(valid_indices))
        action = valid_indices[idx].item()
        
        return action
