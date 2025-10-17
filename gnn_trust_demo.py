"""
Demonstration script for GNN-based trust calculation and task offloading.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.env import Env_Trust
from core.task import Task
from core.vis import *

from zoo.gnn_node import GNNTrustNode
from examples.scenarios.trust_scenario_1 import Scenario

def main():
    # Create the environment with trust scenario
    scenario = Scenario(config_file="examples/scenarios/configs/trust_config_1.json")
    env = Env_Trust(scenario, config_file="core/configs/env_config.json")
    
    # Replace some TrustNodes with GNNTrustNodes
    for node_name in ['n0', 'n1', 'n2']:
        if node_name in env.scenario.node_id2name.values():
            # Get the original node
            original_node = env.scenario.get_node(node_name)
            
            # Create a GNNTrustNode with the same parameters
            gnn_node = GNNTrustNode(
                node_id=original_node.node_id,
                name=original_node.name,
                self_trust=original_node.trust_mat.get(original_node.name, 0.5),
                max_cpu_freq=original_node.max_cpu_freq,
                max_buffer_size=original_node.task_buffer.max_size if hasattr(original_node, 'task_buffer') else 1e6,
                location=getattr(original_node, 'location', None),
                idle_energy_coef=getattr(original_node, 'idle_energy_coef', 0),
                exe_energy_coef=getattr(original_node, 'exe_energy_coef', 0),
                is_malicious=getattr(original_node, 'is_malicious', False),
                gnn_config={
                    'model_type': 'gat',
                    'hidden_dim': 64,
                    'output_dim': 32,
                    'update_frequency': 5
                }
            )
            
            # Replace node in the infrastructure
            env.scenario.infrastructure.remove_node(node_name)
            env.scenario.infrastructure.add_node(gnn_node)
            env.scenario.node_id2name[gnn_node.node_id] = gnn_node.name
    
    # Load simulated tasks
    data = pd.read_csv("examples/dataset/task_dataset.csv")
    simulated_tasks = list(data.iloc[:].values)
    n_tasks = len(simulated_tasks)
    
    # Check the arrival times of tasks for each node
    arrival_times = {node.name: [] for _, node in env.scenario.get_nodes().items()}
    task_assign = {}
    arrival_pointer = {node.name: 0 for _, node in env.scenario.get_nodes().items()}
    
    # The Tasks are already sorted by generation time
    for task_info in simulated_tasks:
        arrival_times[task_info[7]].append(task_info[1])
        task_assign[task_info[2]] = task_info[7]
    
    # Begin Simulation
    until = 1
    launched_tasks = []
    
    for task_info in simulated_tasks:
        # header = ['TaskName', 'GenerationTime', 'TaskID', 'TaskSize', 'CyclesPerBit', 
        #           'TransBitRate', 'DDL', 'SrcName', 'DstName']
        
        # Update time until task generation
        delta_t = task_info[1] - until
        if delta_t > 0:
            # Run simulation for delta_t time
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
        
        # Get source node
        src_node = env.scenario.get_node(task_info[7])
        
        # Use GNN-based node selection if source is GNNTrustNode
        if isinstance(src_node, GNNTrustNode):
            # Get available nodes (excluding source)
            available_nodes = {name: node for name, node in env.scenario.get_nodes().items() 
                              if name != src_node.name}
            
            # Select destination node
            dst_name = src_node.select_node(task, available_nodes)
        else:
            # Use original destination from dataset
            dst_name = task_info[8]
        
        # Execute task
        try:
            env.process(task=task, dst_name=dst_name)
            launched_tasks.append(task.task_id)
            print(f"Task {task.task_id} from {task.src_name} to {dst_name}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Continue simulation until all tasks complete
    while env.task_count < len(launched_tasks):
        env.run(10)
        until += 10
        print(f"Time {until}, Completed {env.task_done} / {len(launched_tasks)} tasks")
    
    # Print success rate
    success_rate = env.task_success / max(1, env.task_count)
    print(f"\nSimulation completed!")
    print(f"Success rate: {success_rate:.2%} ({env.task_success}/{env.task_count})")
    
    # Extract trust scores
    trust_scores = {}
    for name, node in env.scenario.get_nodes().items():
        if isinstance(node, GNNTrustNode):
            # Get trust scores from GNN
            trust_scores[name] = {
                target: node.compute_trust(target)
                for target in env.scenario.get_nodes().keys()
            }
    
    # Print trust scores
    print("\nGNN Trust Scores:")
    for source, targets in trust_scores.items():
        print(f"{source}:")
        for target, score in targets.items():
            print(f"  → {target}: {score:.4f}")
    
    # Visualization: frames to video
    vis_frame2video(env)

if __name__ == '__main__':
    main()