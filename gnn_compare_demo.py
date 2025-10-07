"""
GNN-based trust evaluation and comparison with baseline trust models.
This script demonstrates the advantages of using GNN-based trust for task offloading.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.env import Env_Trust
from core.task import Task
from core.vis import *

from examples.scenarios.gnn_scenario import Scenario
from zoo.node import TrustNode, MaliciousNode
from zoo.gnn_node import GNNTrustNode

def run_experiment(env, tasks, experiment_name="GNN Trust Experiment"):
    """Run an experiment with the given environment and tasks.
    
    Args:
        env: RayCloudSim environment
        tasks: List of tasks to execute
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n=== Starting {experiment_name} ===")
    
    # Reset environment
    env.reset()
    
    # Track metrics
    metrics = {
        'success_rate': 0.0,
        'avg_latency': 0.0,
        'energy_consumption': 0.0,
        'task_distribution': defaultdict(int),
        'execution_time': 0.0,
        'task_success': 0,
        'task_failure': 0,
        'malicious_selections': 0
    }
    
    start_time = time.time()
    until = 0
    launched_tasks = []
    
    # Run tasks
    for i, task_info in enumerate(tasks):
        # Update simulation time
        delta_t = task_info[1] - until
        if delta_t > 0:
            env.run(delta_t)
            until = task_info[1]
        
        # Create task
        task = Task(
            task_id=int(task_info[2]),
            task_size=int(task_info[3]),
            cycles_per_bit=int(task_info[4]),
            trans_bit_rate=int(task_info[5]),
            src_name=task_info[7],
            ddl=int(task_info[6]),
            task_name=task_info[0]
        )
        
        # Track original destination for comparison
        original_dst = task_info[8]
        
        try:
            # Process task (let the environment/policy select the destination)
            env.process(task=task)
            launched_tasks.append(task.task_id)
            
            # Track destination node
            metrics['task_distribution'][task.dst_name] += 1
            
            # Track malicious selections
            dst_node = env.scenario.get_node(task.dst_name)
            if hasattr(dst_node, 'is_malicious') and dst_node.is_malicious:
                metrics['malicious_selections'] += 1
                
        except Exception as e:
            print(f"Error processing task {task.task_id}: {e}")
    
    # Continue simulation until all tasks complete
    while env.task_count < len(launched_tasks):
        env.run(10)
        until += 10
        print(f"Time {until}, Completed {env.task_done} / {len(launched_tasks)} tasks")
    
    # Calculate metrics
    metrics['execution_time'] = time.time() - start_time
    metrics['success_rate'] = env.task_success / max(1, env.task_count)
    metrics['task_success'] = env.task_success
    metrics['task_failure'] = env.task_count - env.task_success
    
    # Calculate average latency
    latencies = []
    for task_id in env.logger.completed_tasks.keys():
        task = env.logger.completed_tasks[task_id]
        latency = task.trans_time + task.wait_time + task.exe_time
        latencies.append(latency)
    
    metrics['avg_latency'] = sum(latencies) / max(1, len(latencies))
    
    # Calculate energy consumption
    metrics['energy_consumption'] = env.scenario.avg_node_energy()
    
    # Print metrics
    print(f"\n=== {experiment_name} Results ===")
    print(f"Success rate: {metrics['success_rate']:.2%} ({metrics['task_success']}/{env.task_count})")
    print(f"Average latency: {metrics['avg_latency']:.2f}")
    print(f"Energy consumption: {metrics['energy_consumption']:.2f}")
    print(f"Malicious selections: {metrics['malicious_selections']}")
    print(f"Execution time: {metrics['execution_time']:.2f}s")
    
    return metrics

def compare_trust_models():
    """Compare different trust models."""
    # Load simulated tasks
    data = pd.read_csv("examples/dataset/task_dataset.csv")
    tasks = list(data.iloc[:].values)
    
    # Create scenarios with different trust models
    scenarios = {
        'GNN Trust': Scenario(config_file="examples/scenarios/configs/gnn_trust_config.json"),
        'Baseline Trust': Scenario(config_file="examples/scenarios/configs/trust_config_1.json")
    }
    
    # Run experiments
    results = {}
    for name, scenario in scenarios.items():
        # Create environment
        env = Env_Trust(scenario, config_file="core/configs/env_config.json")
        
        # Run experiment
        results[name] = run_experiment(env, tasks, f"{name} Model")
    
    # Compare results
    print("\n=== Model Comparison ===")
    for metric in ['success_rate', 'avg_latency', 'energy_consumption', 'malicious_selections']:
        print(f"\n{metric}:")
        for model, metrics in results.items():
            print(f"  {model}: {metrics[metric]}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Success Rate
    plt.subplot(2, 2, 1)
    plt.bar(results.keys(), [metrics['success_rate'] for metrics in results.values()])
    plt.title('Success Rate')
    plt.ylim(0, 1)
    
    # Average Latency
    plt.subplot(2, 2, 2)
    plt.bar(results.keys(), [metrics['avg_latency'] for metrics in results.values()])
    plt.title('Average Latency (lower is better)')
    
    # Energy Consumption
    plt.subplot(2, 2, 3)
    plt.bar(results.keys(), [metrics['energy_consumption'] for metrics in results.values()])
    plt.title('Energy Consumption (lower is better)')
    
    # Malicious Selections
    plt.subplot(2, 2, 4)
    plt.bar(results.keys(), [metrics['malicious_selections'] for metrics in results.values()])
    plt.title('Malicious Node Selections (lower is better)')
    
    plt.tight_layout()
    plt.savefig('logs/trust_model_comparison.png')
    plt.close()
    
    return results

def analyze_gnn_trust_behavior():
    """Analyze the behavior of GNN-based trust over time."""
    # Load tasks
    data = pd.read_csv("examples/dataset/task_dataset.csv")
    tasks = list(data.iloc[:].values)
    
    # Create scenario and environment
    scenario = Scenario(config_file="examples/scenarios/configs/gnn_trust_config.json")
    env = Env_Trust(scenario, config_file="core/configs/env_config.json")
    
    # Track trust evolution
    trust_evolution = defaultdict(lambda: defaultdict(list))
    timestamps = []
    until = 0
    
    # Run tasks and record trust scores
    for i, task_info in enumerate(tasks):
        # Update simulation time
        delta_t = task_info[1] - until
        if delta_t > 0:
            env.run(delta_t)
            until = task_info[1]
        
        # Create task
        task = Task(
            task_id=int(task_info[2]),
            task_size=int(task_info[3]),
            cycles_per_bit=int(task_info[4]),
            trans_bit_rate=int(task_info[5]),
            src_name=task_info[7],
            ddl=int(task_info[6]),
            task_name=task_info[0]
        )
        
        try:
            # Process task
            env.process(task=task)
            
            # Record timestamp
            timestamps.append(until)
            
            # Record trust scores
            for src_name, src_node in scenario.get_nodes().items():
                if isinstance(src_node, GNNTrustNode):
                    for dst_name, dst_node in scenario.get_nodes().items():
                        if src_name != dst_name:
                            trust_score = src_node.compute_trust(dst_name)
                            trust_evolution[src_name][dst_name].append(trust_score)
        
        except Exception as e:
            print(f"Error processing task {task.task_id}: {e}")
    
    # Continue simulation until all tasks complete
    while env.task_count < env.task_done:
        env.run(10)
        until += 10
    
    # Plot trust evolution for GNN nodes
    plt.figure(figsize=(15, 10))
    gnn_nodes = [name for name, node in scenario.get_nodes().items() 
                if isinstance(node, GNNTrustNode)]
    
    for i, gnn_node in enumerate(gnn_nodes):
        plt.subplot(len(gnn_nodes), 1, i+1)
        
        # Plot trust scores for all destinations
        for dst_name, scores in trust_evolution[gnn_node].items():
            # Check if destination is malicious
            dst_node = scenario.get_node(dst_name)
            is_malicious = hasattr(dst_node, 'is_malicious') and dst_node.is_malicious
            
            # Use different line style for malicious nodes
            linestyle = '--' if is_malicious else '-'
            
            plt.plot(timestamps[:len(scores)], scores, label=dst_name, linestyle=linestyle)
        
        plt.title(f'Trust Evolution from {gnn_node}')
        plt.ylabel('Trust Score')
        plt.legend()
        
        if i == len(gnn_nodes) - 1:
            plt.xlabel('Simulation Time')
    
    plt.tight_layout()
    plt.savefig('logs/trust_evolution.png')
    plt.close()

def main():
    """Main function."""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Compare trust models
    compare_results = compare_trust_models()
    
    # Analyze GNN trust behavior
    analyze_gnn_trust_behavior()
    
    print("\nExperiments completed! Results saved in the logs directory.")

if __name__ == '__main__':
    main()