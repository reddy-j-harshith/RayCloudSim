"""
Simple GNN Trust demonstration that tests the basic functionality.
"""
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.env import Env_Trust
from core.task import Task
from examples.scenarios.gnn_scenario import Scenario

def main():
    print("=== GNN Trust Simple Demo ===")
    
    # Create the environment with GNN scenario
    scenario = Scenario(config_file="examples/scenarios/configs/gnn_trust_config.json")
    env = Env_Trust(scenario, config_file="core/configs/gnn_env_config.json")
    
    print(f"Scenario loaded with {len(env.scenario.get_nodes())} nodes")
    
    # Set infrastructure for all GNNTrustNodes
    from zoo.gnn_node import GNNTrustNode
    for name, node in env.scenario.get_nodes().items():
        if isinstance(node, GNNTrustNode):
            node.set_infrastructure(env.scenario.infrastructure)
    
    # Create a simple task
    task = Task(
        task_id=1,
        task_size=100,
        cycles_per_bit=5,
        trans_bit_rate=50,
        src_name='n0',
        ddl=100,
        task_name='test_task'
    )
    
    # Get source node
    src_node = env.scenario.get_node('n0')
    print(f"Source node type: {type(src_node).__name__}")
    
    # Test the node selection if it's a GNNTrustNode
    from zoo.gnn_node import GNNTrustNode
    if isinstance(src_node, GNNTrustNode):
        print("Testing GNN-based node selection...")
        
        # Get available nodes (excluding source)
        available_nodes = {name: node for name, node in env.scenario.get_nodes().items() 
                          if name != src_node.name}
        
        print(f"Available destination nodes: {list(available_nodes.keys())}")
        
        # Select destination node
        dst_name = src_node.select_node(task, available_nodes)
        print(f"GNN selected destination: {dst_name}")
        
        # Test trust computation
        if dst_name:
            trust_score = src_node.compute_trust(dst_name)
            print(f"Trust score for {dst_name}: {trust_score:.4f}")
    else:
        print("Source node is not a GNNTrustNode. Using default destination.")
        dst_name = 'n1'
    
    # Execute the task
    try:
        env.process(task=task, dst_name=dst_name)
        print(f"Task {task.task_id} processed from {task.src_name} to {dst_name}")
        
        # Run simulation until completion
        env.run(200)
        
        # Print results
        print(f"Total tasks processed: {env.task_count}")
        print(f"Completed tasks: {len(env.done_task_info)}")
        print(f"Task completed successfully: {len(env.done_task_info) > 0}")
        
    except Exception as e:
        import traceback
        print(f"Error processing task: {e}")
        print("Full traceback:")
        traceback.print_exc()
    
    print("\n=== Demo completed ===")

if __name__ == '__main__':
    main()