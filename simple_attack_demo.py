#!/usr/bin/env python3
"""
Simplified Attack Simulation Demo
Demonstrates the attack simulation framework without complex dependencies.
"""

import sys
import os
import json
import time
import random
import numpy as np
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath('.'))

from core.task import Task
from zoo.node import TrustNode, MaliciousNode


class SimpleScenario:
    """Simplified scenario for attack simulation demo."""
    
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.nodes = {}
        self._create_nodes(config['Nodes'])
        self._create_links(config['Links'])
    
    def _create_nodes(self, node_configs):
        """Create nodes from configuration."""
        for node_config in node_configs:
            node_type = node_config.get('NodeType', 'TrustNode')
            node_name = node_config['NodeName']
            
            if node_type == 'MaliciousNode':
                node = MaliciousNode(
                    node_id=node_config['NodeID'],
                    name=node_name,
                    self_trust=node_config.get('SelfTrust', 0.5),
                    mal_type=0,
                    max_cpu_freq=node_config.get('MaxFreq', 1000)
                )
            else:  # TrustNode or GNNTrustNode (simplified as TrustNode)
                node = TrustNode(
                    node_id=node_config['NodeID'], 
                    name=node_name,
                    self_trust=node_config.get('SelfTrust', 0.5),
                    max_cpu_freq=node_config.get('MaxFreq', 1000)
                )
            
            self.nodes[node_name] = node
    
    def _create_links(self, link_configs):
        """Create links between nodes (simplified)."""
        # Just track connectivity for trust relationships
        for link_config in link_configs:
            from_node = link_config['FromNodeName']
            to_node = link_config['ToNodeName']
            
            if from_node in self.nodes and to_node in self.nodes:
                # Initialize trust relationships
                self.nodes[from_node].trust_mat[to_node] = 0.5
                self.nodes[to_node].trust_mat[from_node] = 0.5
    
    def get_nodes(self):
        """Get all nodes."""
        return self.nodes
    
    def get_node(self, name):
        """Get specific node."""
        return self.nodes.get(name)


class SimpleAttackSimulator:
    """Simplified attack simulator."""
    
    def __init__(self, scenario):
        self.scenario = scenario
        self.time_step = 0
        self.attack_history = []
    
    def simulate_on_off_attack(self, malicious_nodes: List[str]):
        """Simulate on-off attacks."""
        for node_name in malicious_nodes:
            node = self.scenario.get_node(node_name)
            if isinstance(node, MaliciousNode):
                node.cycle_behavior()
        
        self.attack_history.append({
            'type': 'on_off',
            'timestamp': self.time_step,
            'nodes': malicious_nodes
        })
    
    def simulate_ballot_stuffing_attack(self, malicious_nodes: List[str], target_nodes: List[str]):
        """Simulate ballot stuffing attacks."""
        for attacker_name in malicious_nodes:
            attacker = self.scenario.get_node(attacker_name)
            if isinstance(attacker, MaliciousNode):
                for target_name in target_nodes:
                    if target_name != attacker_name:
                        attacker.trust_mat[target_name] = 0.9  # Fake high trust
        
        self.attack_history.append({
            'type': 'ballot_stuffing',
            'timestamp': self.time_step,
            'attackers': malicious_nodes,
            'targets': target_nodes
        })
    
    def simulate_bad_mouthing_attack(self, malicious_nodes: List[str], target_nodes: List[str]):
        """Simulate bad-mouthing attacks."""
        for attacker_name in malicious_nodes:
            attacker = self.scenario.get_node(attacker_name)
            if isinstance(attacker, MaliciousNode):
                for target_name in target_nodes:
                    if target_name != attacker_name:
                        attacker.trust_mat[target_name] = 0.1  # Fake low trust
        
        self.attack_history.append({
            'type': 'bad_mouthing',
            'timestamp': self.time_step,
            'attackers': malicious_nodes,
            'targets': target_nodes
        })
    
    def advance_time(self):
        """Advance time step."""
        self.time_step += 1


class SimpleTaskAllocator:
    """Simplified task allocation strategies."""
    
    @staticmethod
    def trust_based_allocation(src_node, available_nodes: List[str]) -> str:
        """Allocate based on trust scores."""
        if not available_nodes:
            return None
        
        best_node = None
        best_trust = -1
        
        for node_name in available_nodes:
            trust_score = src_node.trust_mat.get(node_name, 0.5)
            if trust_score > best_trust:
                best_trust = trust_score
                best_node = node_name
        
        return best_node or available_nodes[0]
    
    @staticmethod
    def random_allocation(available_nodes: List[str]) -> str:
        """Random allocation."""
        return random.choice(available_nodes) if available_nodes else None


def create_demo_config():
    """Create demo configuration."""
    config = {
        "Nodes": [
            # Malicious nodes
            {"NodeID": 0, "NodeName": "mal1", "NodeType": "MaliciousNode", "MaxFreq": 1000, "SelfTrust": 0.8},
            {"NodeID": 1, "NodeName": "mal2", "NodeType": "MaliciousNode", "MaxFreq": 1200, "SelfTrust": 0.7},
            {"NodeID": 2, "NodeName": "mal3", "NodeType": "MaliciousNode", "MaxFreq": 1100, "SelfTrust": 0.9},
            
            # Honest nodes
            {"NodeID": 3, "NodeName": "honest1", "NodeType": "TrustNode", "MaxFreq": 1300, "SelfTrust": 0.9},
            {"NodeID": 4, "NodeName": "honest2", "NodeType": "TrustNode", "MaxFreq": 1200, "SelfTrust": 0.8},
            {"NodeID": 5, "NodeName": "honest3", "NodeType": "TrustNode", "MaxFreq": 1400, "SelfTrust": 0.9},
            {"NodeID": 6, "NodeName": "honest4", "NodeType": "TrustNode", "MaxFreq": 1100, "SelfTrust": 0.8}
        ],
        "Links": [
            {"FromNodeName": "mal1", "ToNodeName": "honest1", "BandWidth": 100},
            {"FromNodeName": "mal1", "ToNodeName": "honest2", "BandWidth": 100},
            {"FromNodeName": "mal2", "ToNodeName": "honest3", "BandWidth": 100},
            {"FromNodeName": "mal2", "ToNodeName": "honest4", "BandWidth": 100},
            {"FromNodeName": "mal3", "ToNodeName": "honest1", "BandWidth": 100},
            {"FromNodeName": "honest1", "ToNodeName": "honest2", "BandWidth": 100},
            {"FromNodeName": "honest2", "ToNodeName": "honest3", "BandWidth": 100},
            {"FromNodeName": "honest3", "ToNodeName": "honest4", "BandWidth": 100}
        ]
    }
    
    config_path = "simple_demo_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def demo_attack_scenarios():
    """Demo different attack scenarios."""
    print("=== Simplified Attack Simulation Demo ===")
    
    # Create demo configuration
    config_path = create_demo_config()
    
    # Create scenario
    scenario = SimpleScenario(config_path)
    
    # Identify node types
    malicious_nodes = []
    honest_nodes = []
    
    for name, node in scenario.get_nodes().items():
        if isinstance(node, MaliciousNode):
            malicious_nodes.append(name)
        else:
            honest_nodes.append(name)
    
    print(f"Malicious nodes: {malicious_nodes}")
    print(f"Honest nodes: {honest_nodes}")
    
    # Initialize attack simulator
    attack_simulator = SimpleAttackSimulator(scenario)
    
    # Run attack scenarios
    attack_scenarios = [
        ('on_off', 'On-Off Attack'),
        ('ballot_stuffing', 'Ballot Stuffing Attack'),
        ('bad_mouthing', 'Bad-Mouthing Attack')
    ]
    
    results = {}
    
    for attack_type, attack_name in attack_scenarios:
        print(f"\n--- {attack_name} ---")
        
        # Reset scenario for each attack
        scenario = SimpleScenario(config_path)
        attack_simulator = SimpleAttackSimulator(scenario)
        
        # Initialize metrics
        successful_tasks = 0
        failed_tasks = 0
        tasks_to_malicious = 0 
        tasks_to_honest = 0
        
        # Simulate over time
        for time_step in range(100):
            attack_simulator.time_step = time_step
            
            # Execute attack every 20 time steps
            if time_step % 20 == 0 and time_step > 0:
                if attack_type == 'on_off':
                    attack_simulator.simulate_on_off_attack(malicious_nodes)
                elif attack_type == 'ballot_stuffing':
                    attack_simulator.simulate_ballot_stuffing_attack(malicious_nodes, honest_nodes)
                elif attack_type == 'bad_mouthing':
                    attack_simulator.simulate_bad_mouthing_attack(malicious_nodes, honest_nodes)
            
            # Generate tasks every 5 time steps
            if time_step % 5 == 0:
                task_id = time_step // 5 + 1
                
                # Select source node (from honest nodes)
                src_name = random.choice(honest_nodes)
                src_node = scenario.get_node(src_name)
                
                # Get available destination nodes
                available_nodes = [name for name in scenario.get_nodes().keys() if name != src_name]
                
                # Allocate task using trust-based strategy
                dst_name = SimpleTaskAllocator.trust_based_allocation(src_node, available_nodes)
                
                if dst_name:
                    # Simulate task execution
                    dst_node = scenario.get_node(dst_name)
                    
                    # Create simple task
                    task = type('Task', (), {
                        'task_id': task_id,
                        'task_size': 100,
                        'timestamp': time_step
                    })()
                    
                    # Execute task
                    if isinstance(dst_node, MaliciousNode):
                        # Malicious node may fail or succeed based on current behavior
                        success = dst_node.perform_task(task)
                        tasks_to_malicious += 1
                    else:
                        # Honest node usually succeeds
                        success = random.random() > 0.1  # 90% success rate
                        tasks_to_honest += 1
                    
                    if success:
                        successful_tasks += 1
                        # Update trust positively
                        src_node.update_trust(dst_name, True)
                    else:
                        failed_tasks += 1
                        # Update trust negatively
                        src_node.update_trust(dst_name, False)
            
            attack_simulator.advance_time()
        
        # Calculate results
        total_tasks = successful_tasks + failed_tasks
        success_rate = successful_tasks / max(1, total_tasks)
        malicious_task_ratio = tasks_to_malicious / max(1, successful_tasks + failed_tasks)
        
        results[attack_type] = {
            'success_rate': success_rate,
            'malicious_task_ratio': malicious_task_ratio,
            'total_tasks': total_tasks,
            'tasks_to_malicious': tasks_to_malicious,
            'tasks_to_honest': tasks_to_honest
        }
        
        print(f"Total tasks: {total_tasks}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Tasks to malicious nodes: {tasks_to_malicious} ({malicious_task_ratio:.2%})")
        print(f"Tasks to honest nodes: {tasks_to_honest}")
        
        # Show some trust scores
        print("Final trust scores (sample):")
        sample_honest = honest_nodes[0]
        sample_node = scenario.get_node(sample_honest)
        for target_name in list(scenario.get_nodes().keys())[:3]:
            if target_name != sample_honest:
                trust_score = sample_node.trust_mat.get(target_name, 0.5)
                node_type = "Malicious" if target_name in malicious_nodes else "Honest"
                print(f"  {sample_honest} -> {target_name} ({node_type}): {trust_score:.3f}")
    
    # Clean up
    try:
        os.remove(config_path)
    except:
        pass
    
    return results


def demo_trust_evolution():
    """Demonstrate trust score evolution during attacks."""
    print("\n=== Trust Evolution Demo ===")
    
    config_path = create_demo_config()
    scenario = SimpleScenario(config_path) 
    attack_simulator = SimpleAttackSimulator(scenario)
    
    # Get node lists
    malicious_nodes = [name for name, node in scenario.get_nodes().items() 
                      if isinstance(node, MaliciousNode)]
    honest_nodes = [name for name, node in scenario.get_nodes().items() 
                   if not isinstance(node, MaliciousNode)]
    
    # Track trust evolution
    trust_history = {}
    
    # Sample node to track
    tracker_node = honest_nodes[0]
    tracker = scenario.get_node(tracker_node)
    
    print(f"Tracking trust scores from {tracker_node}")
    
    for time_step in range(50):
        attack_simulator.time_step = time_step
        
        # Execute combined attacks
        if time_step % 15 == 0 and time_step > 0:
            attack_simulator.simulate_ballot_stuffing_attack(malicious_nodes, honest_nodes)
            attack_simulator.simulate_bad_mouthing_attack(malicious_nodes[:1], honest_nodes)
            print(f"Time {time_step}: Attacks executed")
        
        # Simulate some interactions
        if time_step % 3 == 0:
            for target_name in list(scenario.get_nodes().keys())[:4]:
                if target_name != tracker_node:
                    # Simulate interaction result
                    if target_name in malicious_nodes:
                        # Malicious nodes more likely to fail
                        success = random.random() > 0.6
                    else:
                        # Honest nodes more likely to succeed
                        success = random.random() > 0.2
                    
                    tracker.update_trust(target_name, success)
        
        # Record trust scores
        if time_step % 5 == 0:
            trust_scores = {}
            for target_name in scenario.get_nodes().keys():
                if target_name != tracker_node:
                    trust_scores[target_name] = tracker.trust_mat.get(target_name, 0.5)
            trust_history[time_step] = trust_scores.copy()
        
        attack_simulator.advance_time()
    
    # Show trust evolution
    print("\nTrust score evolution:")
    print("Time\t" + "\t".join(f"{name[:6]}" for name in sorted(trust_history[0].keys())))
    
    for time_step in sorted(trust_history.keys()):
        scores = trust_history[time_step]
        score_str = "\t".join(f"{scores[name]:.3f}" for name in sorted(scores.keys()))
        print(f"{time_step}\t{score_str}")
    
    # Show attack statistics for malicious nodes
    print("\nMalicious node attack statistics:")
    for node_name in malicious_nodes:
        node = scenario.get_node(node_name)
        if isinstance(node, MaliciousNode):
            stats = node.get_attack_statistics()
            print(f"{node_name}: {stats['malicious_interactions']}/{stats['total_interactions']} "
                  f"malicious ({stats['malicious_ratio']:.2%})")
    
    # Clean up
    try:
        os.remove(config_path)
    except:
        pass


def main():
    """Main demo function."""
    print("=== Comprehensive Attack Simulation Demo ===")
    print("This demo shows various attack scenarios and their impact on trust-based task allocation.")
    
    # Run attack scenario demos
    results = demo_attack_scenarios()
    
    # Run trust evolution demo
    demo_trust_evolution()
    
    # Summary
    print("\n=== Demo Summary ===")
    print("Attack Impact Analysis:")
    
    for attack_type, metrics in results.items():
        attack_name = attack_type.replace('_', ' ').title()
        print(f"\n{attack_name}:")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Tasks to Malicious: {metrics['malicious_task_ratio']:.2%}")
        print(f"  Total Tasks: {metrics['total_tasks']}")
    
    # Identify most/least effective attacks
    by_malicious_ratio = sorted(results.items(), key=lambda x: x[1]['malicious_task_ratio'], reverse=True)
    
    print(f"\nMost effective attack (highest malicious task ratio): {by_malicious_ratio[0][0].replace('_', ' ').title()}")
    print(f"Least effective attack (lowest malicious task ratio): {by_malicious_ratio[-1][0].replace('_', ' ').title()}")
    
    print("\n=== Key Insights ===")
    print("1. Trust-based allocation can adapt to attacks over time")
    print("2. Different attacks have varying levels of effectiveness")
    print("3. Trust scores evolve based on interaction outcomes")
    print("4. Malicious nodes show different behavior patterns during attacks")
    
    print("\nDemo completed successfully!")


if __name__ == '__main__':
    main()