#!/usr/bin/env python3
"""Debug version of GNN Trust Simple Demo to see what's happening with node selection."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from core.env import Env_Trust
from core.task import Task
from examples.scenarios.gnn_scenario import Scenario

def main():
    print("=== GNN Trust Debug Demo ===")
    
    # Create the environment with GNN scenario
    scenario = Scenario(config_file="examples/scenarios/configs/gnn_trust_config.json")
    env = Env_Trust(scenario, config_file="core/configs/env_config.json")
    
    print(f"Scenario loaded with {len(env.scenario.get_nodes())} nodes")
    
    # Set infrastructure for all GNNTrustNodes
    from zoo.gnn_node import GNNTrustNode
    for name, node in env.scenario.get_nodes().items():
        if isinstance(node, GNNTrustNode):
            node.set_infrastructure(env.scenario.infrastructure)
            print(f"Set infrastructure for {name}")
    
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
    if isinstance(src_node, GNNTrustNode):
        print("Testing GNN-based node selection...")
        
        # Get available nodes (excluding source)
        available_nodes = {name: node for name, node in env.scenario.get_nodes().items() 
                          if name != src_node.name}
        
        print(f"Available destination nodes: {list(available_nodes.keys())}")
        
        # Add debug information to the GNN policy
        policy = src_node.gnn_trust_policy
        print(f"Policy config: {policy.config}")
        
        # Check the graph
        graph = policy._get_graph()
        print(f"Graph nodes: {list(graph.nodes())}")
        print(f"Graph edges: {list(graph.edges())}")
        
        # Debug the infrastructure links
        links = env.scenario.infrastructure.get_links()
        print(f"Links type: {type(links)}")
        if isinstance(links, dict):
            print(f"Infrastructure links (dict keys): {list(links.keys())}")
            if links:
                print(f"Sample link: {next(iter(links.values()))}")
        elif isinstance(links, list):
            print(f"Infrastructure links (list): {links}")
        else:
            print(f"Infrastructure links (other): {links}")
            
        # Try to manually add edges
        if isinstance(links, list):
            graph = policy._get_graph()
            print(f"Before manual add - Graph edges: {list(graph.edges())}")
            for link_info in links:
                if len(link_info) >= 2:
                    src, dst = link_info[0], link_info[1]
                    graph.add_edge(src, dst)
            print(f"After manual add - Graph edges: {list(graph.edges())}")
            policy.graph_cache = graph  # Update the cache
        
        # Check trust scores manually
        policy._update_graph(graph, available_nodes)
        policy._calculate_trust_scores(graph, available_nodes.keys())
        print(f"Trust scores: {policy.trust_scores}")
        
        # Check threshold calculation
        task_context = {
            'criticality': getattr(task, 'criticality', 0.5),
            'qos_requirement': 1.0 if getattr(task, 'ddl', -1) > 0 else 0.5
        }
        network_state = {
            'congestion': policy._estimate_congestion(available_nodes),
            'attack_detected': policy._detect_attacks()
        }
        threshold = policy.threshold.compute_threshold(task_context, network_state)
        print(f"Computed threshold: {threshold}")
        
        # Now try selection
        dst_name = src_node.select_node(task, available_nodes)
        print(f"GNN selected destination: {dst_name}")
        
    print("=== Debug completed ===")

if __name__ == "__main__":
    main()