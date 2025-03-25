"""
Test the trained GCNN branching policy on MIP instances.
"""

import os
import sys
import torch
import numpy as np
from pyscipopt import Model
import time
import random

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl4mip.envs.branching_env import MIPBranchingEnv
from rl4mip.models.policies.basic_policies import RandomBranchingPolicy
from rl4mip.models.policies.gcnn_policy import GCNNBranchingPolicy


def create_mip_model(instance_path):
    """Create a MIP model from an instance file.
    
    Args:
        instance_path: Path to the instance file
        
    Returns:
        Model: SCIP model
    """
    scip = Model()
    scip.readProblem(instance_path)
    return scip


def print_results(name, result):
    """Print the results of solving an instance.
    
    Args:
        name: Name of the policy
        result: Dictionary with solving results
    """
    print(f"\n=== Results for {name} ===")
    print(f"Status: {result['status']}")
    print(f"Time: {result['time']:.2f} seconds")
    print(f"Nodes: {result['nodes']}")
    print(f"LP iterations: {result['lp_iterations']}")
    
    if result['status'] != "infeasible":
        print(f"Objective value: {result['primal_bound']:.6f}")
        print(f"Best bound: {result['dual_bound']:.6f}")
        print(f"Gap: {result['gap'] * 100:.6f}%")
    
    if 'branching_decisions' in result:
        print(f"Branching decisions: {result['branching_decisions']}")
    
    if 'primal_dual_integral' in result and result['primal_dual_integral'] is not None:
        print(f"Primal-dual integral: {result['primal_dual_integral']:.6f}")


def test_model_on_instance(instance_path, model_path=None, seed=42):
    """Test a trained GCNN model and baselines on an instance.
    
    Args:
        instance_path: Path to the instance file
        model_path: Path to the trained model
        seed: Random seed for reproducibility
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    instance_name = os.path.basename(instance_path)
    print(f"\n===== Testing on instance: {instance_name} =====")
    
    # Create model generator function
    def model_generator():
        return create_mip_model(instance_path)
    
    # Create environment
    env = MIPBranchingEnv(problem_generator=model_generator)
    
    # Initialize policies
    policies = {
        "Random": RandomBranchingPolicy(),
        "GCNN": GCNNBranchingPolicy(pretrained_path=model_path),
        "SCIP Default": None
    }
    
    results = {}
    
    # Test each policy
    for name, policy in policies.items():
        print(f"\nTesting {name} policy...")
        
        # Solve with policy
        start_time = time.time()
        result = env.reset(policy=policy)
        solve_time = time.time() - start_time
        
        # Store results
        results[name] = result
        
        # Print results
        print_results(name, result)
    
    # Print comparison
    print("\n===== Performance Comparison =====")
    for name, result in results.items():
        print(f"{name}: {result['nodes']} nodes, {result['lp_iterations']} LP iterations, {result['time']:.2f}s")
    
    # Calculate improvements over SCIP Default
    if "SCIP Default" in results and "GCNN" in results:
        scip_result = results["SCIP Default"]
        gcnn_result = results["GCNN"]
        
        # Only compare if both solved to optimality
        if scip_result["status"] == "optimal" and gcnn_result["status"] == "optimal":
            node_reduction = (1 - gcnn_result["nodes"] / scip_result["nodes"]) * 100 if scip_result["nodes"] > 0 else 0
            time_reduction = (1 - gcnn_result["time"] / scip_result["time"]) * 100 if scip_result["time"] > 0 else 0
            lp_iter_reduction = (1 - gcnn_result["lp_iterations"] / scip_result["lp_iterations"]) * 100 if scip_result["lp_iterations"] > 0 else 0
            
            print("\n===== GCNN vs SCIP Default =====")
            print(f"Node reduction: {node_reduction:.2f}%")
            print(f"Time reduction: {time_reduction:.2f}%")
            print(f"LP iterations reduction: {lp_iter_reduction:.2f}%")
    
    return results


def main():
    """Test the trained GCNN model on all instances."""
    # Set random seed
    seed = 42
    
    # Load trained model if available
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "gcnn_imitation.pt")
    
    if not os.path.exists(model_path):
        print(f"Warning: Trained model not found at {model_path}")
        print("Testing with untrained model instead")
        model_path = None
    else:
        print(f"Found trained model at {model_path}")
    
    # Get all instance files
    instance_dir = os.path.join(os.path.dirname(__file__), "instances")
    instance_files = [f for f in os.listdir(instance_dir) if f.endswith(".lp")]
    
    # Sort to ensure deterministic order
    instance_files.sort()
    
    all_results = {}
    
    # Test on each instance
    for instance_file in instance_files:
        instance_path = os.path.join(instance_dir, instance_file)
        results = test_model_on_instance(instance_path, model_path, seed)
        all_results[instance_file] = results
    
    # Print overall summary
    print("\n===== Overall Summary =====")
    
    # Calculate average improvements
    avg_node_reduction = 0
    avg_time_reduction = 0
    avg_lp_iter_reduction = 0
    count = 0
    
    for instance_file, results in all_results.items():
        if "SCIP Default" in results and "GCNN" in results:
            scip_result = results["SCIP Default"]
            gcnn_result = results["GCNN"]
            
            # Only compare if both solved to optimality
            if scip_result["status"] == "optimal" and gcnn_result["status"] == "optimal":
                node_reduction = (1 - gcnn_result["nodes"] / scip_result["nodes"]) * 100 if scip_result["nodes"] > 0 else 0
                time_reduction = (1 - gcnn_result["time"] / scip_result["time"]) * 100 if scip_result["time"] > 0 else 0
                lp_iter_reduction = (1 - gcnn_result["lp_iterations"] / scip_result["lp_iterations"]) * 100 if scip_result["lp_iterations"] > 0 else 0
                
                avg_node_reduction += node_reduction
                avg_time_reduction += time_reduction
                avg_lp_iter_reduction += lp_iter_reduction
                count += 1
    
    if count > 0:
        avg_node_reduction /= count
        avg_time_reduction /= count
        avg_lp_iter_reduction /= count
        
        print(f"Average node reduction: {avg_node_reduction:.2f}%")
        print(f"Average time reduction: {avg_time_reduction:.2f}%")
        print(f"Average LP iterations reduction: {avg_lp_iter_reduction:.2f}%")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()