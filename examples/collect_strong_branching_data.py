"""
Collect strong branching decisions from MIP instances for imitation learning.
"""

import os
import sys
import pickle
import torch
import numpy as np
from pyscipopt import Model, Branchrule, SCIP_RESULT

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class StrongBranchingCollector(Branchrule):
    """SCIP branching rule that collects data from strong branching."""
    
    def __init__(self, scip):
        """Initialize the collector."""
        self.scip = scip
        self.states = []
        self.actions = []
        self.branching_calls = 0
    
    def branchinit(self):
        """Reset statistics when branching begins."""
        self.branching_calls = 0
        self.states = []
        self.actions = []
    
    def branchexeclp(self, allowaddcons):
        """Collect strong branching decisions."""
        try:
            self.branching_calls += 1
            
            # Get branching candidates
            candidates, cand_sols, cand_fracs, ncands, npriocands, _ = self.scip.getLPBranchCands()
            
            if ncands == 0:
                return {"result": SCIP_RESULT.DIDNOTRUN}
            
            # Get bipartite graph representation
            col_features, edge_features, row_features, feature_maps = self.scip.getBipartiteGraphRepresentation(
                prev_col_features=None, 
                prev_edge_features=None, 
                prev_row_features=None,
                suppress_warnings=True
            )
            
            # Get LP columns 
            lp_cols = self.scip.getLPColsData()
            
            # Create mapping from variable to column index
            var_to_col_idx = {}
            
            for i, col in enumerate(lp_cols):
                var = col.getVar()
                var_to_col_idx[var.name] = i
            
            # Step 1: Map candidate variables to their column indices
            candidates_with_idx = []
            for i in range(npriocands):
                var = candidates[i]
                if var.name in var_to_col_idx:
                    col_idx = var_to_col_idx[var.name]
                    candidates_with_idx.append((var, col_idx))
            
            # Step 2: Create boolean mask for candidates
            candidate_mask = [0] * len(col_features)
            for var, col_idx in candidates_with_idx:
                candidate_mask[col_idx] = 1
            
            # Create state
            state = {
                "col_features": col_features,
                "row_features": row_features,
                "edge_features": edge_features,
                "branchable_mask": candidate_mask
            }
            
            # Start strong branching
            self.scip.startStrongbranch()
            
            best_score = float('-inf')
            best_col_idx = None  # Column index of the best variable
            
            # Current LP objective value
            current_bound = self.scip.getLPObjVal()
            
            # Check if the objective is maximizing
            is_maximizing = (self.scip.getObjectiveSense() == "maximize")
            
            # Evaluate each candidate with strong branching
            for var, col_idx in candidates_with_idx:
                # Use strong branching to evaluate the variable
                down, up, downvalid, upvalid, downinf, upinf, _, _, _ = self.scip.getVarStrongbranch(var, 1000, True)
                
                # Skip invalid or infeasible branches
                if not downvalid or not upvalid or downinf or upinf:
                    continue
                
                # Calculate gains based on objective direction
                if is_maximizing:
                    down_gain = down - current_bound
                    up_gain = up - current_bound
                else:
                    down_gain = current_bound - down
                    up_gain = current_bound - up
                
                # Use SCIP's built-in score function
                score = self.scip.getBranchScoreMultiple(var, [down_gain, up_gain])
                
                if score > best_score:
                    best_score = score
                    best_col_idx = col_idx
            
            # End strong branching
            self.scip.endStrongbranch()
            
            # Store the expert decision if valid
            if best_col_idx is not None:
                # Make sure there's a valid mapping to index the state features
                if best_col_idx < len(col_features):
                    # Store the state and action
                    self.states.append(state)
                    self.actions.append(best_col_idx)
                    
                    # Find the variable corresponding to the best column index
                    best_var = None
                    for var, col_idx in candidates_with_idx:
                        if col_idx == best_col_idx:
                            best_var = var
                            break
                    
                    if best_var:
                        # Branch on the best variable
                        self.scip.branchVar(best_var)
                        
                        if self.branching_calls % 10 == 0:
                            print(f"Collected {len(self.states)} strong branching samples")
                        
                        return {"result": SCIP_RESULT.BRANCHED}
            
            return {"result": SCIP_RESULT.DIDNOTRUN}
            
        except Exception as e:
            print(f"Error in data collection: {e}")
            return {"result": SCIP_RESULT.DIDNOTRUN}


def collect_data_from_instance(instance_path):
    """Collect data from a single instance.
    
    Args:
        instance_path: Path to the MIP instance file
        
    Returns:
        tuple: (states, actions) collected from the instance
    """
    print(f"Collecting data from {instance_path}")
    
    # Create SCIP model
    scip = Model()
    scip.readProblem(instance_path)
    
    # Set SCIP parameters
    scip.setIntParam("display/verblevel", 0)  # Reduce output
    
    # Create and include strong branching collector
    collector = StrongBranchingCollector(scip)
    scip.includeBranchrule(
        collector, "strong_branching_collector", "Collects strong branching decisions",
        priority=9999999, maxdepth=-1, maxbounddist=1
    )
    
    # Solve the problem
    try:
        scip.optimize()
        print(f"  Status: {scip.getStatus()}")
        print(f"  Collected {len(collector.states)} samples")
        return collector.states, collector.actions
    except Exception as e:
        print(f"  Error during optimization: {e}")
        return [], []


def main():
    """Collect data from all instances and save to file."""
    # Get all instance files
    instance_dir = os.path.join(os.path.dirname(__file__), "instances")
    instance_files = [f for f in os.listdir(instance_dir) if f.endswith(".lp")]
    
    # Sort to ensure deterministic order
    instance_files.sort()
    
    all_states = []
    all_actions = []
    
    # Collect data from each instance
    for instance_file in instance_files:
        instance_path = os.path.join(instance_dir, instance_file)
        states, actions = collect_data_from_instance(instance_path)
        all_states.extend(states)
        all_actions.extend(actions)
    
    # Save collected data
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "strong_branching_data.pkl"), "wb") as f:
        pickle.dump({
            "states": all_states,
            "actions": all_actions
        }, f)
    
    print(f"\nTotal samples collected: {len(all_states)}")
    print(f"Data saved to {os.path.join(output_dir, 'strong_branching_data.pkl')}")


if __name__ == "__main__":
    main()