from pyscipopt import Model
import torch
import numpy as np

from rl4mip.envs.branching_env import MIPBranchingEnv
from rl4mip.models.policies.basic_policies import RandomBranchingPolicy

def create_mip_model():
    """Create or load a MIP instance"""
    scip = Model()
    # Change this path to the location of your MIP instance
    scip.readProblem("./examples/instances/instance_2.lp")
    return scip

def test_rl_branching():
    """Test RL-based branching against SCIP default"""
    # Create environment and policy
    env = MIPBranchingEnv(problem_generator=create_mip_model)
    policy = RandomBranchingPolicy()
    
    # Run with policy
    print("\n=== Testing with RL Policy ===")
    results_with_policy = env.reset(policy=policy)
    
    # Run without policy (SCIP default)
    print("\n=== Testing with SCIP Default ===")
    results_without_policy = env.reset(policy=None)
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"RL Policy: {results_with_policy['nodes']} nodes, {results_with_policy['branching_decisions']} decisions")
    print(f"SCIP Default: {results_without_policy['nodes']} nodes")

if __name__ == "__main__":
    test_rl_branching()
