#!/usr/bin/env python3
"""
Comprehensive Attack Simulation Demo
Demonstrates the attack simulation framework with various attack scenarios.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath('.'))

from attack_simulation import (
    ComprehensiveAttackExperiment, 
    AttackSimulator, 
    TaskAllocationStrategy,
    TrustEvaluationMetrics
)
from core.env import Env_Trust
from core.task import Task
from examples.scenarios.gnn_scenario import Scenario
from zoo.gnn_node import GNNTrustNode
from zoo.node import TrustNode, MaliciousNode


def create_demo_config():
    """Create a demo configuration for attack simulation."""
    config = {
        "Nodes": [
            # Malicious nodes (first 5)
            {"NodeID": 0, "NodeName": "n0", "NodeType": "MaliciousNode", "MaxFreq": 1000, "SelfTrust": 0.8},
            {"NodeID": 1, "NodeName": "n1", "NodeType": "MaliciousNode", "MaxFreq": 1200, "SelfTrust": 0.7},
            {"NodeID": 2, "NodeName": "n2", "NodeType": "MaliciousNode", "MaxFreq": 1100, "SelfTrust": 0.9},
            {"NodeID": 3, "NodeName": "n3", "NodeType": "MaliciousNode", "MaxFreq": 1300, "SelfTrust": 0.6},
            {"NodeID": 4, "NodeName": "n4", "NodeType": "MaliciousNode", "MaxFreq": 1000, "SelfTrust": 0.8},
            
            # GNN trust nodes (next 3)
            {"NodeID": 5, "NodeName": "n5", "NodeType": "GNNTrustNode", "MaxFreq": 1500, "SelfTrust": 0.9},
            {"NodeID": 6, "NodeName": "n6", "NodeType": "GNNTrustNode", "MaxFreq": 1400, "SelfTrust": 0.8},
            {"NodeID": 7, "NodeName": "n7", "NodeType": "GNNTrustNode", "MaxFreq": 1600, "SelfTrust": 0.9},
            
            # Honest trust nodes (remaining)
            {"NodeID": 8, "NodeName": "n8", "NodeType": "TrustNode", "MaxFreq": 1200, "SelfTrust": 0.9},
            {"NodeID": 9, "NodeName": "n9", "NodeType": "TrustNode", "MaxFreq": 1300, "SelfTrust": 0.8},
            {"NodeID": 10, "NodeName": "n10", "NodeType": "TrustNode", "MaxFreq": 1100, "SelfTrust": 0.9},
            {"NodeID": 11, "NodeName": "n11", "NodeType": "TrustNode", "MaxFreq": 1400, "SelfTrust": 0.8},
            {"NodeID": 12, "NodeName": "n12", "NodeType": "TrustNode", "MaxFreq": 1000, "SelfTrust": 0.9}
        ],
        "Links": [
            {"FromNodeName": "n0", "ToNodeName": "n1", "BandWidth": 100},
            {"FromNodeName": "n0", "ToNodeName": "n5", "BandWidth": 100},
            {"FromNodeName": "n1", "ToNodeName": "n2", "BandWidth": 100},
            {"FromNodeName": "n1", "ToNodeName": "n6", "BandWidth": 100},
            {"FromNodeName": "n2", "ToNodeName": "n3", "BandWidth": 100},
            {"FromNodeName": "n2", "ToNodeName": "n7", "BandWidth": 100},
            {"FromNodeName": "n3", "ToNodeName": "n4", "BandWidth": 100},
            {"FromNodeName": "n3", "ToNodeName": "n8", "BandWidth": 100},
            {"FromNodeName": "n4", "ToNodeName": "n0", "BandWidth": 100},
            {"FromNodeName": "n4", "ToNodeName": "n9", "BandWidth": 100},
            {"FromNodeName": "n5", "ToNodeName": "n6", "BandWidth": 100},
            {"FromNodeName": "n5", "ToNodeName": "n10", "BandWidth": 100},
            {"FromNodeName": "n6", "ToNodeName": "n7", "BandWidth": 100},
            {"FromNodeName": "n6", "ToNodeName": "n11", "BandWidth": 100},
            {"FromNodeName": "n7", "ToNodeName": "n8", "BandWidth": 100},
            {"FromNodeName": "n7", "ToNodeName": "n12", "BandWidth": 100},
            {"FromNodeName": "n8", "ToNodeName": "n9", "BandWidth": 100},
            {"FromNodeName": "n9", "ToNodeName": "n10", "BandWidth": 100},
            {"FromNodeName": "n10", "ToNodeName": "n11", "BandWidth": 100},
            {"FromNodeName": "n11", "ToNodeName": "n12", "BandWidth": 100},
            {"FromNodeName": "n12", "ToNodeName": "n5", "BandWidth": 100}
        ]
    }
    
    config_path = "demo_attack_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def demo_individual_attacks():
    """Demonstrate individual attack scenarios."""
    print("=== Individual Attack Scenarios Demo ===")
    
    # Create demo configuration
    config_path = create_demo_config()
    
    # Individual attacks to test
    attacks = ['on_off', 'ballot_stuffing', 'bad_mouthing', 'collusion', 'sybil']
    strategies = ['gnn_trust', 'random', 'greedy']
    
    results = {}
    
    for attack_type in attacks:
        print(f"\n--- Testing {attack_type} attack ---")
        
        attack_results = {}
        
        for strategy in strategies:
            print(f"Strategy: {strategy}")
            
            # Create scenario
            scenario = Scenario(config_file=config_path)
            
            # Create simple env config
            env_config = {
                "Basic": {"VisFrame": "off", "Train": "off", "Test": "off"},
                "VisFrame": {"LogInfoPath": "logs/demo", "LogFramesPath": "logs/demo/frames", "TargetNodeList": []},
                "Train": {"CloseLogger": "True"},
                "Test": {}
            }
            
            env_config_path = "demo_env_config.json"
            with open(env_config_path, 'w') as f:
                json.dump(env_config, f, indent=2)
            
            env = Env_Trust(scenario, config_file=env_config_path)
            
            # Set up GNN nodes
            for name, node in env.scenario.get_nodes().items():
                if isinstance(node, GNNTrustNode):
                    node.set_infrastructure(env.scenario.infrastructure)
            
            # Initialize attack simulator
            attack_simulator = AttackSimulator(scenario, {})
            
            # Identify node types
            malicious_nodes = []
            honest_nodes = []
            gnn_nodes = []
            
            for name, node in scenario.get_nodes().items():
                if isinstance(node, MaliciousNode):
                    malicious_nodes.append(name)
                elif isinstance(node, GNNTrustNode):
                    gnn_nodes.append(name)
                else:
                    honest_nodes.append(name)
            
            # Initialize trust metrics
            trust_metrics = TrustEvaluationMetrics()
            
            # Simulate attack scenario
            successful_tasks = 0
            failed_tasks = 0
            tasks_to_malicious = 0
            
            simulation_time = 200
            num_tasks = 20
            
            for time_step in range(0, simulation_time, 10):
                attack_simulator.time_step = time_step
                
                # Execute specific attack at certain intervals
                if time_step % 50 == 0 and time_step > 0:
                    if attack_type == 'on_off':
                        attack_simulator.simulate_on_off_attack(malicious_nodes, [10], [15])
                    elif attack_type == 'ballot_stuffing':
                        attack_simulator.simulate_ballot_stuffing_attack(
                            malicious_nodes, honest_nodes + gnn_nodes, 0.9)
                    elif attack_type == 'bad_mouthing':
                        attack_simulator.simulate_bad_mouthing_attack(
                            malicious_nodes, honest_nodes + gnn_nodes, 0.1)
                    elif attack_type == 'collusion':
                        if len(malicious_nodes) >= 2:
                            attack_simulator.simulate_collusion_attack(malicious_nodes[:2])
                    elif attack_type == 'sybil':
                        if len(malicious_nodes) >= 3:
                            attack_simulator.simulate_sybil_attack(malicious_nodes[1:], malicious_nodes[0])
                
                # Generate tasks
                if time_step % 10 == 0 and time_step // 10 < num_tasks:
                    task_id = time_step // 10 + 1
                    task = Task(
                        task_id=task_id,
                        task_size=100 + np.random.randint(50, 150),
                        cycles_per_bit=5,
                        trans_bit_rate=50,
                        src_name=np.random.choice(gnn_nodes + honest_nodes),
                        ddl=time_step + 50,
                        task_name=f'demo_task_{task_id}'
                    )
                    
                    # Select destination based on strategy
                    src_node = scenario.get_node(task.src_name)
                    available_nodes = {name: node for name, node in scenario.get_nodes().items() 
                                     if name != task.src_name}
                    
                    if strategy == 'gnn_trust':
                        dst_name = TaskAllocationStrategy.gnn_trust_based(src_node, available_nodes, task)
                    elif strategy == 'random':
                        dst_name = TaskAllocationStrategy.random_allocation(list(available_nodes.keys()))
                    else:  # greedy
                        dst_name = TaskAllocationStrategy.greedy_resources(available_nodes)
                    
                    # Process task
                    try:
                        env.process(task=task, dst_name=dst_name)
                        successful_tasks += 1
                        
                        if dst_name in malicious_nodes:
                            tasks_to_malicious += 1
                            
                    except Exception as e:
                        failed_tasks += 1
                        print(f"Task {task_id} failed: {e}")
                
                # Record trust scores
                if time_step % 20 == 0:
                    trust_scores = {}
                    ground_truth = {}
                    
                    for name, node in scenario.get_nodes().items():
                        if isinstance(node, (GNNTrustNode, TrustNode)):
                            for target_name in scenario.get_nodes().keys():
                                if target_name != name:
                                    trust_key = f"{name}->{target_name}"
                                    if isinstance(node, GNNTrustNode):
                                        trust_scores[trust_key] = node.compute_trust(target_name)
                                    else:
                                        trust_scores[trust_key] = node.trust_mat.get(target_name, 0.5)
                                    ground_truth[trust_key] = target_name not in malicious_nodes
                    
                    trust_metrics.record_trust_scores(time_step, trust_scores, ground_truth)
                
                attack_simulator.advance_time()
            
            # Calculate results
            total_tasks = successful_tasks + failed_tasks
            success_rate = successful_tasks / max(1, total_tasks)
            malicious_task_ratio = tasks_to_malicious / max(1, successful_tasks)
            
            # Get classification metrics
            classification_metrics = trust_metrics.compute_classification_metrics()
            
            attack_results[strategy] = {
                'success_rate': success_rate,
                'malicious_task_ratio': malicious_task_ratio,
                'total_tasks': total_tasks,
                'metrics': classification_metrics
            }
            
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Tasks to malicious: {malicious_task_ratio:.2%}")
            if classification_metrics:
                print(f"  Trust accuracy: {classification_metrics.get('accuracy', 0):.3f}")
                print(f"  AUC: {classification_metrics.get('auc', 0):.3f}")
        
        results[attack_type] = attack_results
    
    # Clean up
    try:
        os.remove(config_path)
        os.remove(env_config_path)
    except:
        pass
    
    return results


def demo_combined_attacks():
    """Demonstrate combined attack scenarios."""
    print("\n=== Combined Attack Scenarios Demo ===")
    
    # Create demo configuration
    config_path = create_demo_config()
    
    # Combined attack scenarios
    combined_attacks = [
        ['on_off', 'ballot_stuffing'],
        ['bad_mouthing', 'collusion'],
        ['sybil', 'ballot_stuffing']
    ]
    
    results = {}
    
    for attacks in combined_attacks:
        attack_name = '_'.join(attacks)
        print(f"\n--- Testing combined {attack_name} attacks ---")
        
        # Create scenario
        scenario = Scenario(config_file=config_path)
        
        # Create env config
        env_config = {
            "Basic": {"VisFrame": "off", "Train": "off", "Test": "off"},
            "VisFrame": {"LogInfoPath": "logs/demo", "LogFramesPath": "logs/demo/frames", "TargetNodeList": []},
            "Train": {"CloseLogger": "True"},
            "Test": {}
        }
        
        env_config_path = "demo_env_config.json"
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f, indent=2)
        
        env = Env_Trust(scenario, config_file=env_config_path)
        
        # Set up GNN nodes
        for name, node in env.scenario.get_nodes().items():
            if isinstance(node, GNNTrustNode):
                node.set_infrastructure(env.scenario.infrastructure)
        
        # Initialize attack simulator
        attack_simulator = AttackSimulator(scenario, {})
        
        # Identify node types
        malicious_nodes = []
        honest_nodes = []
        gnn_nodes = []
        
        for name, node in scenario.get_nodes().items():
            if isinstance(node, MaliciousNode):
                malicious_nodes.append(name)
            elif isinstance(node, GNNTrustNode):
                gnn_nodes.append(name)
            else:
                honest_nodes.append(name)
        
        # Initialize trust metrics
        trust_metrics = TrustEvaluationMetrics()
        
        # Test GNN trust strategy
        successful_tasks = 0
        failed_tasks = 0
        tasks_to_malicious = 0
        
        simulation_time = 300
        num_tasks = 30
        
        for time_step in range(0, simulation_time, 10):
            attack_simulator.time_step = time_step
            
            # Execute combined attacks at intervals
            if time_step % 40 == 0 and time_step > 0:
                for attack_type in attacks:
                    if attack_type == 'on_off':
                        attack_simulator.simulate_on_off_attack(malicious_nodes, [8], [12])
                    elif attack_type == 'ballot_stuffing':
                        attack_simulator.simulate_ballot_stuffing_attack(
                            malicious_nodes, honest_nodes + gnn_nodes, 0.9)
                    elif attack_type == 'bad_mouthing':
                        attack_simulator.simulate_bad_mouthing_attack(
                            malicious_nodes, honest_nodes + gnn_nodes, 0.1)
                    elif attack_type == 'collusion':
                        if len(malicious_nodes) >= 2:
                            attack_simulator.simulate_collusion_attack(malicious_nodes[:2])
                    elif attack_type == 'sybil':
                        if len(malicious_nodes) >= 3:
                            attack_simulator.simulate_sybil_attack(malicious_nodes[1:], malicious_nodes[0])
            
            # Generate tasks
            if time_step % 10 == 0 and time_step // 10 < num_tasks:
                task_id = time_step // 10 + 1
                task = Task(
                    task_id=task_id,
                    task_size=100 + np.random.randint(50, 150),
                    cycles_per_bit=5,
                    trans_bit_rate=50,
                    src_name=np.random.choice(gnn_nodes + honest_nodes),
                    ddl=time_step + 50,
                    task_name=f'combined_task_{task_id}'
                )
                
                # Use GNN trust strategy
                src_node = scenario.get_node(task.src_name)
                available_nodes = {name: node for name, node in scenario.get_nodes().items() 
                                 if name != task.src_name}
                
                dst_name = TaskAllocationStrategy.gnn_trust_based(src_node, available_nodes, task)
                
                # Process task
                try:
                    env.process(task=task, dst_name=dst_name)
                    successful_tasks += 1
                    
                    if dst_name in malicious_nodes:
                        tasks_to_malicious += 1
                        
                except Exception as e:
                    failed_tasks += 1
                    print(f"Task {task_id} failed: {e}")
            
            # Record trust scores
            if time_step % 20 == 0:
                trust_scores = {}
                ground_truth = {}
                
                for name, node in scenario.get_nodes().items():
                    if isinstance(node, (GNNTrustNode, TrustNode)):
                        for target_name in scenario.get_nodes().keys():
                            if target_name != name:
                                trust_key = f"{name}->{target_name}"
                                if isinstance(node, GNNTrustNode):
                                    trust_scores[trust_key] = node.compute_trust(target_name)
                                else:
                                    trust_scores[trust_key] = node.trust_mat.get(target_name, 0.5)
                                ground_truth[trust_key] = target_name not in malicious_nodes
                
                trust_metrics.record_trust_scores(time_step, trust_scores, ground_truth)
            
            attack_simulator.advance_time()
        
        # Calculate results
        total_tasks = successful_tasks + failed_tasks
        success_rate = successful_tasks / max(1, total_tasks)
        malicious_task_ratio = tasks_to_malicious / max(1, successful_tasks)
        
        # Get classification metrics
        classification_metrics = trust_metrics.compute_classification_metrics()
        
        results[attack_name] = {
            'success_rate': success_rate,
            'malicious_task_ratio': malicious_task_ratio,
            'total_tasks': total_tasks,
            'metrics': classification_metrics
        }
        
        print(f"Success rate: {success_rate:.2%}")
        print(f"Tasks to malicious: {malicious_task_ratio:.2%}")
        if classification_metrics:
            print(f"Trust accuracy: {classification_metrics.get('accuracy', 0):.3f}")
            print(f"AUC: {classification_metrics.get('auc', 0):.3f}")
            print(f"F1-macro: {classification_metrics.get('f1_macro', 0):.3f}")
    
    # Clean up
    try:
        os.remove(config_path)
        os.remove(env_config_path)
    except:
        pass
    
    return results


def create_summary_visualization(individual_results, combined_results):
    """Create summary visualization of demo results."""
    print("\n--- Creating Summary Visualizations ---")
    
    # Create plots directory
    os.makedirs('demo_plots', exist_ok=True)
    
    # Plot 1: Individual Attack Impact Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success rates for individual attacks
    attack_types = list(individual_results.keys())
    gnn_success = [individual_results[attack]['gnn_trust']['success_rate'] for attack in attack_types]
    random_success = [individual_results[attack]['random']['success_rate'] for attack in attack_types]
    greedy_success = [individual_results[attack]['greedy']['success_rate'] for attack in attack_types]
    
    x = np.arange(len(attack_types))
    width = 0.25
    
    axes[0,0].bar(x - width, gnn_success, width, label='GNN Trust', color='blue', alpha=0.7)
    axes[0,0].bar(x, random_success, width, label='Random', color='red', alpha=0.7)
    axes[0,0].bar(x + width, greedy_success, width, label='Greedy', color='green', alpha=0.7)
    
    axes[0,0].set_xlabel('Attack Type')
    axes[0,0].set_ylabel('Success Rate')
    axes[0,0].set_title('Success Rate by Attack Type and Strategy')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels([a.replace('_', ' ').title() for a in attack_types], rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Malicious task ratios for individual attacks
    gnn_malicious = [individual_results[attack]['gnn_trust']['malicious_task_ratio'] for attack in attack_types]
    random_malicious = [individual_results[attack]['random']['malicious_task_ratio'] for attack in attack_types]
    greedy_malicious = [individual_results[attack]['greedy']['malicious_task_ratio'] for attack in attack_types]
    
    axes[0,1].bar(x - width, gnn_malicious, width, label='GNN Trust', color='blue', alpha=0.7)
    axes[0,1].bar(x, random_malicious, width, label='Random', color='red', alpha=0.7)
    axes[0,1].bar(x + width, greedy_malicious, width, label='Greedy', color='green', alpha=0.7)
    
    axes[0,1].set_xlabel('Attack Type')
    axes[0,1].set_ylabel('Malicious Task Ratio')
    axes[0,1].set_title('Malicious Task Ratio by Attack Type and Strategy')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels([a.replace('_', ' ').title() for a in attack_types], rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Combined attacks comparison
    combined_attack_types = list(combined_results.keys())
    combined_success = [combined_results[attack]['success_rate'] for attack in combined_attack_types]
    combined_malicious = [combined_results[attack]['malicious_task_ratio'] for attack in combined_attack_types]
    
    axes[1,0].bar(range(len(combined_attack_types)), combined_success, color='purple', alpha=0.7)
    axes[1,0].set_xlabel('Combined Attack Type')
    axes[1,0].set_ylabel('Success Rate')
    axes[1,0].set_title('GNN Trust Performance Under Combined Attacks')
    axes[1,0].set_xticks(range(len(combined_attack_types)))
    axes[1,0].set_xticklabels([a.replace('_', ' ').title() for a in combined_attack_types], rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].bar(range(len(combined_attack_types)), combined_malicious, color='orange', alpha=0.7)
    axes[1,1].set_xlabel('Combined Attack Type')
    axes[1,1].set_ylabel('Malicious Task Ratio')
    axes[1,1].set_title('Malicious Task Ratio Under Combined Attacks')
    axes[1,1].set_xticks(range(len(combined_attack_types)))
    axes[1,1].set_xticklabels([a.replace('_', ' ').title() for a in combined_attack_types], rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_plots/attack_demo_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Trust Metrics Comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Extract trust metrics for comparison
    metrics_data = []
    for attack_type in individual_results:
        for strategy in individual_results[attack_type]:
            metrics = individual_results[attack_type][strategy]['metrics']
            if metrics:
                metrics_data.append({
                    'Attack': attack_type.replace('_', ' ').title(),
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Accuracy': metrics.get('accuracy', 0),
                    'AUC': metrics.get('auc', 0),
                    'F1_Macro': metrics.get('f1_macro', 0)
                })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        
        # Create grouped bar chart for accuracy
        strategies = df['Strategy'].unique()
        attacks = df['Attack'].unique()
        
        x = np.arange(len(attacks))
        width = 0.25
        
        for i, strategy in enumerate(strategies):
            strategy_data = df[df['Strategy'] == strategy]
            accuracies = [strategy_data[strategy_data['Attack'] == attack]['Accuracy'].iloc[0] 
                         if len(strategy_data[strategy_data['Attack'] == attack]) > 0 else 0 
                         for attack in attacks]
            
            ax.bar(x + i * width, accuracies, width, label=strategy)
        
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Trust Prediction Accuracy')
        ax.set_title('Trust Prediction Accuracy by Attack Type and Strategy')
        ax.set_xticks(x + width)
        ax.set_xticklabels(attacks, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demo_plots/trust_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Summary visualizations saved to demo_plots/")


def main():
    """Main demo function."""
    print("=== Comprehensive Attack Simulation Demo ===")
    print("This demo shows the attack simulation framework capabilities.")
    
    # Run individual attack demos
    individual_results = demo_individual_attacks()
    
    # Run combined attack demos
    combined_results = demo_combined_attacks()
    
    # Create summary visualizations
    create_summary_visualization(individual_results, combined_results)
    
    # Print summary
    print("\n=== Demo Summary ===")
    print("\nIndividual Attack Results:")
    for attack_type, strategies in individual_results.items():
        print(f"\n{attack_type.replace('_', ' ').title()} Attack:")
        for strategy, results in strategies.items():
            print(f"  {strategy.replace('_', ' ').title()}: "
                  f"Success={results['success_rate']:.2%}, "
                  f"Malicious={results['malicious_task_ratio']:.2%}")
    
    print("\nCombined Attack Results:")
    for attack_name, results in combined_results.items():
        print(f"{attack_name.replace('_', ' ').title()}: "
              f"Success={results['success_rate']:.2%}, "
              f"Malicious={results['malicious_task_ratio']:.2%}")
    
    # Show key insights
    print("\n=== Key Insights ===")
    
    # Best performing strategy
    avg_gnn_success = np.mean([individual_results[attack]['gnn_trust']['success_rate'] 
                              for attack in individual_results])
    avg_random_success = np.mean([individual_results[attack]['random']['success_rate'] 
                                 for attack in individual_results])
    avg_greedy_success = np.mean([individual_results[attack]['greedy']['success_rate'] 
                                 for attack in individual_results])
    
    print(f"Average success rates across all individual attacks:")
    print(f"  GNN Trust: {avg_gnn_success:.2%}")
    print(f"  Random: {avg_random_success:.2%}")
    print(f"  Greedy: {avg_greedy_success:.2%}")
    
    # Most challenging attacks
    gnn_malicious_ratios = {attack: individual_results[attack]['gnn_trust']['malicious_task_ratio'] 
                           for attack in individual_results}
    worst_attack = max(gnn_malicious_ratios, key=gnn_malicious_ratios.get)
    best_attack = min(gnn_malicious_ratios, key=gnn_malicious_ratios.get)
    
    print(f"\nMost challenging attack for GNN Trust: {worst_attack.replace('_', ' ').title()} "
          f"({gnn_malicious_ratios[worst_attack]:.2%} malicious tasks)")
    print(f"Best defended attack: {best_attack.replace('_', ' ').title()} "
          f"({gnn_malicious_ratios[best_attack]:.2%} malicious tasks)")
    
    print(f"\nDemo completed! Check 'demo_plots/' for visualizations.")


if __name__ == '__main__':
    main()