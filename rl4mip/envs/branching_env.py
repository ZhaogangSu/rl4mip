import torch
import numpy as np
from pyscipopt import Model, Branchrule, SCIP_RESULT

class RLBranchRule(Branchrule):
    """SCIP branching rule that uses a provided policy for decisions"""
    
    def __init__(self, scip, policy=None):
        self.scip = scip
        self.policy = policy
        self.branching_calls = 0
        self.branching_decisions = 0
        self.selected_vars = []
        self.states = []
        self.actions = []
    
    def branchinit(self):
        """Reset statistics when branching begins"""
        self.branching_calls = 0
        self.branching_decisions = 0
        self.selected_vars = []
        self.states = []
        self.actions = []
    
    def branchexeclp(self, allowaddcons):
        """Branching callback - called when LP solution is optimal but fractional"""
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
            
            # Get LP columns and create mapping
            lp_cols = self.scip.getLPColsData()
            
            # Create mappings
            var_name_to_col_idx = {}
            col_idx_to_candidate = {}
            
            for i, col in enumerate(lp_cols):
                var = col.getVar()
                var_name_to_col_idx[var.name] = i
            
            # Create branchable mask
            candidate_mask = [0] * len(col_features)
            
            # Mark branchable columns
            for i in range(npriocands):
                var = candidates[i]
                if var.name in var_name_to_col_idx:
                    col_idx = var_name_to_col_idx[var.name]
                    if col_idx < len(candidate_mask):
                        candidate_mask[col_idx] = 1
                        col_idx_to_candidate[col_idx] = var
            
            # Create state for policy
            state = {
                "col_features": torch.tensor(col_features, dtype=torch.float32),
                "edge_features": torch.tensor(edge_features, dtype=torch.float32),
                "row_features": torch.tensor(row_features, dtype=torch.float32),
                "branchable_mask": torch.tensor(candidate_mask, dtype=torch.bool)
            }
            
            # Store state for later learning
            self.states.append(state)
            
            # Check if we can proceed
            if self.policy is None or state['branchable_mask'].sum().item() == 0:
                return {"result": SCIP_RESULT.DIDNOTRUN}
            
            # Use policy to select column
            with torch.no_grad():
                col_action = self.policy.act(state)
            
            if col_action is None or col_action not in col_idx_to_candidate:
                return {"result": SCIP_RESULT.DIDNOTRUN}
            
            # Get selected variable and branch
            selected_var = col_idx_to_candidate[col_action]
            down_child, eq_child, up_child = self.scip.branchVarVal(
                selected_var, selected_var.getLPSol()
            )
            
            # Record decision
            self.actions.append(col_action)
            self.selected_vars.append(selected_var)
            self.branching_decisions += 1
            
            if self.branching_decisions % 100 == 0:
                print(f"Executed {self.branching_decisions}-th branch decision")
            
            return {"result": SCIP_RESULT.BRANCHED}
            
        except Exception as e:
            print(f"Error in branching rule: {e}")
            return {"result": SCIP_RESULT.DIDNOTRUN}

class MIPBranchingEnv:
    """Environment for training/evaluating MIP branching policies"""
    
    def __init__(self, problem_generator=None, instance_path=None):
        self.problem_generator = problem_generator
        self.instance_path = instance_path
        self.scip = None
        self.branch_rule = None
        self.episode_count = 0
    
    def reset(self, policy=None):
        """Create new SCIP instance and solve with the given policy"""
        # Create SCIP model
        if self.problem_generator:
            self.scip = self.problem_generator()
        elif self.instance_path:
            self.scip = Model()
            self.scip.readProblem(self.instance_path)
        else:
            raise ValueError("Either problem_generator or instance_path must be provided")
        
        # Report problem stats
        vars = self.scip.getVars()
        conss = self.scip.getConss()
        print(f"Problem has {len(vars)} variables and {len(conss)} constraints")
        
        # Set SCIP parameters
        self.scip.setIntParam("display/verblevel", 1)
        self.scip.setLongintParam("limits/nodes", 50000)  # Limit nodes for training
        
        # Create branching rule
        self.branch_rule = RLBranchRule(self.scip, policy)
        self.scip.includeBranchrule(
            self.branch_rule, "rl_branching", "RL-based branching rule",
            priority=9999999, maxdepth=-1, maxbounddist=1
        )
        
        # Solve the problem
        print("Starting optimization...")
        try:
            import time
            start_time = time.time()
            self.scip.optimize()
            solve_time = time.time() - start_time
            
            print(f"Optimization finished - Status: {self.scip.getStatus()}")
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            solve_time = 0
        
# Collect optimization results
        status = self.scip.getStatus()
        nodes = self.scip.getNNodes()
        lp_iterations = self.scip.getNLPIterations()
        
        # Objective values and gap
        primal_bound = self.scip.getPrimalbound() if self.scip.getStatus() != "infeasible" else float('inf')
        dual_bound = self.scip.getDualbound()
        gap = self.scip.getGap() if status in ["optimal", "bestsollimit", "nodelimit", "totalnodelimit", "stallnodelimit", "timelimit"] else float('inf')
        
        # Cuts information
        n_cuts_applied = self.scip.getNCutsApplied()
        
        # Get primal-dual integral
        # Use the standalone readStatistics function
        import tempfile
        import os
        from pyscipopt import readStatistics
        
        stats_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stats')
        stats_filename = stats_file.name
        stats_file.close()
        
        self.scip.writeStatistics(stats_filename)
        
        # Read the statistics file to get the primal-dual integral
        try:
            stats = readStatistics(stats_filename)
            primal_dual_integral = stats.primal_dual_integral if stats.primal_dual_integral is not None else float('inf')
        except Exception as e:
            print(f"Warning: Could not read primal-dual integral: {e}")
            primal_dual_integral = None
        
        # Clean up the temporary file
        try:
            os.unlink(stats_filename)
        except:
            pass
        
        # Return enhanced results
        return {
            "status": status,
            "nodes": nodes,
            "time": solve_time,
            "lp_iterations": lp_iterations,
            "primal_bound": primal_bound,
            "dual_bound": dual_bound,
            "gap": gap,
            "primal_dual_integral": primal_dual_integral,
            "cuts_applied": n_cuts_applied,
            "states": self.branch_rule.states,
            "actions": self.branch_rule.actions,
            "branching_calls": self.branch_rule.branching_calls,
            "branching_decisions": self.branch_rule.branching_decisions
        }