# from pyscipopt import Model
# import torch
# import numpy as np

# from rl4mip.envs.branching_env import MIPBranchingEnv
# from rl4mip.models.policies.rand_policy import RandomBranchingPolicy

# def create_mip_model():
#     """Create or load a MIP instance"""
#     scip = Model()
#     # Change this path to the location of your MIP instance
#     scip.readProblem("./examples/instances/instance_2.lp")
#     return scip

# def test_rl_branching():
#     """Test RL-based branching against SCIP default"""
#     # Create environment and policy
#     env = MIPBranchingEnv(problem_generator=create_mip_model)
#     policy = RandomBranchingPolicy()
    
#     # Run with policy
#     print("\n=== Testing with RL Policy ===")
#     results_with_policy = env.reset(policy=policy)
    
#     # Run without policy (SCIP default)
#     print("\n=== Testing with SCIP Default ===")
#     results_without_policy = env.reset(policy=None)
    
#     # Compare results
#     print("\n=== Comparison ===")
#     print(f"RL Policy: {results_with_policy['nodes']} nodes, {results_with_policy['branching_decisions']} decisions")
#     print(f"SCIP Default: {results_without_policy['nodes']} nodes")

# if __name__ == "__main__":
#     test_rl_branching()

"""Test script to compare GCNN policy vs Random policy using enhanced environment metrics."""

from pyscipopt import Model
import torch
import numpy as np

from rl4mip.envs.branching_env import MIPBranchingEnv
from rl4mip.models.policies.rand_policy import RandomBranchingPolicy
from rl4mip.models.policies.gcnn_policy import GCNNBranchingPolicy


def create_mip_model():
    """Create or load the MIP instance."""
    scip = Model()
    # Load the instance file
    scip.readProblem("./examples/instances/instance_2.lp")
    return scip


def print_optimization_results(name, result):
    """Print detailed results from optimization.
    
    Args:
        name: Name of the policy
        result: Result dictionary from the environment
    """
    print(f"\n=== Results from {name} ===")
    print(f"Status: {result['status']}")
    print(f"Time: {result['time']:.2f} seconds")
    print(f"Nodes: {result['nodes']}")
    print(f"LP iterations: {result['lp_iterations']}")
    
    if result['status'] != "infeasible":
        print(f"Objective value: {result['primal_bound']:.6f}")
        print(f"Best bound: {result['dual_bound']:.6f}")
        print(f"Gap: {result['gap'] * 100:.6f}%")
    
    if result['primal_dual_integral'] is not None:
        print(f"Primal-dual integral: {result['primal_dual_integral']:.6f}")
    else:
        print("Primal-dual integral: Not available")
    
    if 'branching_decisions' in result:
        print(f"Branching decisions: {result['branching_decisions']}")
    
    print(f"Cuts applied: {result['cuts_applied']}")
    



def compare_policies():
    """Compare GCNN policy with random policy and SCIP default."""
    # Create environment
    env = MIPBranchingEnv(problem_generator=create_mip_model)
    
    # Initialize policies
    random_policy = RandomBranchingPolicy()
    gcnn_policy = GCNNBranchingPolicy()  # Untrained for now
    
    print("\n===== Running experiments with different policies =====")
    
    # Test random policy
    print("\n=== Testing with Random Policy ===")
    random_result = env.reset(policy=random_policy)
    print_optimization_results("Random Policy", random_result)
    
    # Test GCNN policy
    print("\n=== Testing with GCNN Policy (untrained) ===")
    gcnn_result = env.reset(policy=gcnn_policy)
    print_optimization_results("GCNN Policy", gcnn_result)
    
    # Test SCIP default
    print("\n=== Testing with SCIP Default ===")
    default_result = env.reset(policy=None)
    print_optimization_results("SCIP Default", default_result)
    
    # Compare results
    print("\n===== Results Summary =====")
    print(f"Random Policy: {random_result['nodes']} nodes, {random_result['lp_iterations']} LP iterations, {random_result['time']:.2f}s")
    print(f"GCNN Policy: {gcnn_result['nodes']} nodes, {gcnn_result['lp_iterations']} LP iterations, {gcnn_result['time']:.2f}s")
    print(f"SCIP Default: {default_result['nodes']} nodes, {default_result['lp_iterations']} LP iterations, {default_result['time']:.2f}s")
    
    if random_result['status'] == gcnn_result['status'] == default_result['status'] == "optimal":
        print("\nAll policies found the optimal solution:")
        print(f"Objective value: {random_result['primal_bound']:.6f}")


if __name__ == "__main__":
    compare_policies()