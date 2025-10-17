#!/usr/bin/env python3
"""
Attack Simulation Framework for GNN Trust System
Implements various attack scenarios and comprehensive evaluation metrics.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, f1_score, accuracy_score,
    mean_absolute_error, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath('.'))

from core.env import Env_Trust
from core.task import Task
from examples.scenarios.gnn_scenario import Scenario
from zoo.gnn_node import GNNTrustNode
from zoo.node import TrustNode, MaliciousNode


class AttackSimulator:
    """Simulates various types of attacks in the trust network."""
    
    def __init__(self, scenario, attack_config: Dict[str, Any]):
        self.scenario = scenario
        self.attack_config = attack_config
        self.attack_history = []
        self.time_step = 0
        
    def simulate_on_off_attack(self, malicious_nodes: List[str], 
                              good_periods: List[int], bad_periods: List[int]):
        """Simulate on-off attacks where malicious nodes alternate behavior."""
        attack_info = {
            'type': 'on_off',
            'timestamp': self.time_step,
            'nodes': malicious_nodes,
            'good_periods': good_periods,
            'bad_periods': bad_periods
        }
        
        for i, node_name in enumerate(malicious_nodes):
            node = self.scenario.get_node(node_name)
            if isinstance(node, MaliciousNode):
                # Determine current behavior based on time cycles
                cycle_length = good_periods[i % len(good_periods)] + bad_periods[i % len(bad_periods)]
                position_in_cycle = self.time_step % cycle_length
                
                if position_in_cycle < good_periods[i % len(good_periods)]:
                    # Good behavior period
                    node.is_behaving_well = True
                    node.attack_probaboility = 0.1
                else:
                    # Bad behavior period  
                    node.is_behaving_well = False
                    node.attack_probability = 0.9
                    
        self.attack_history.append(attack_info)
        
    def simulate_ballot_stuffing_attack(self, malicious_nodes: List[str], 
                                       target_nodes: List[str], fake_trust: float = 0.9):
        """Simulate ballot stuffing where malicious nodes give false positive ratings."""
        attack_info = {
            'type': 'ballot_stuffing',
            'timestamp': self.time_step,
            'attackers': malicious_nodes,
            'targets': target_nodes,
            'fake_trust': fake_trust
        }
        
        for attacker_name in malicious_nodes:
            attacker = self.scenario.get_node(attacker_name)
            if isinstance(attacker, (MaliciousNode, TrustNode)):
                for target_name in target_nodes:
                    if target_name != attacker_name:
                        # Artificially inflate trust scores
                        attacker.trust_mat[target_name] = fake_trust
                        
        self.attack_history.append(attack_info)
        
    def simulate_bad_mouthing_attack(self, malicious_nodes: List[str], 
                                    target_nodes: List[str], fake_trust: float = 0.1):
        """Simulate bad-mouthing where malicious nodes give false negative ratings."""
        attack_info = {
            'type': 'bad_mouthing',
            'timestamp': self.time_step,
            'attackers': malicious_nodes,
            'targets': target_nodes,
            'fake_trust': fake_trust
        }
        
        for attacker_name in malicious_nodes:
            attacker = self.scenario.get_node(attacker_name)
            if isinstance(attacker, (MaliciousNode, TrustNode)):
                for target_name in target_nodes:
                    if target_name != attacker_name:
                        # Artificially deflate trust scores
                        attacker.trust_mat[target_name] = fake_trust
                        
        self.attack_history.append(attack_info)
        
    def simulate_collusion_attack(self, colluding_nodes: List[str], 
                                 target_nodes: List[str] = None):
        """Simulate collusion where multiple nodes coordinate attacks."""
        attack_info = {
            'type': 'collusion',
            'timestamp': self.time_step,
            'colluding_nodes': colluding_nodes,
            'target_nodes': target_nodes or []
        }
        
        # Colluding nodes boost each other's trust
        for i, node1_name in enumerate(colluding_nodes):
            node1 = self.scenario.get_node(node1_name)
            if isinstance(node1, (MaliciousNode, TrustNode)):
                for j, node2_name in enumerate(colluding_nodes):
                    if i != j:
                        node1.trust_mat[node2_name] = 0.95  # High mutual trust
                        
                # Attack target nodes if specified
                if target_nodes:
                    for target_name in target_nodes:
                        if target_name not in colluding_nodes:
                            node1.trust_mat[target_name] = 0.1  # Low trust for targets
                            
        self.attack_history.append(attack_info)
        
    def simulate_sybil_attack(self, sybil_nodes: List[str], controller_node: str):
        """Simulate Sybil attack where one entity controls multiple identities."""
        attack_info = {
            'type': 'sybil',
            'timestamp': self.time_step,
            'sybil_nodes': sybil_nodes,
            'controller': controller_node
        }
        
        controller = self.scenario.get_node(controller_node)
        if isinstance(controller, (MaliciousNode, TrustNode)):
            controller_trust = controller.trust_mat.copy()
            
            # All Sybil nodes mimic controller behavior
            for sybil_name in sybil_nodes:
                sybil_node = self.scenario.get_node(sybil_name)
                if isinstance(sybil_node, (MaliciousNode, TrustNode)):
                    sybil_node.trust_mat = controller_trust.copy()
                    sybil_node.is_sybil = True
                    sybil_node.controller = controller_node
                    
        self.attack_history.append(attack_info)
        
    def advance_time(self):
        """Advance simulation time step."""
        self.time_step += 1


class TaskAllocationStrategy:
    """Different task allocation strategies for comparison."""
    
    @staticmethod
    def round_robin(available_nodes: List[str], task_counter: int) -> str:
        """Round-robin allocation."""
        return available_nodes[task_counter % len(available_nodes)]
    
    @staticmethod
    def random_allocation(available_nodes: List[str]) -> str:
        """Random allocation."""
        return np.random.choice(available_nodes)
    
    @staticmethod
    def greedy_resources(available_nodes: Dict[str, Any]) -> str:
        """Greedy allocation based on available resources."""
        best_node = None
        best_score = -1
        
        for node_name, node in available_nodes.items():
            # Simple resource score: CPU frequency + buffer availability
            resource_score = (node.free_cpu_freq / max(1, node.max_cpu_freq) + 
                            (1 - len(node.task_buffer.task_ids) / max(1, node.task_buffer.max_size)))
            
            if resource_score > best_score:
                best_score = resource_score
                best_node = node_name
                
        return best_node or list(available_nodes.keys())[0]
    
    @staticmethod
    def gnn_trust_based(src_node, available_nodes: Dict[str, Any], task) -> str:
        """GNN-based trust allocation."""
        if isinstance(src_node, GNNTrustNode):
            return src_node.select_node(task, available_nodes)
        else:
            # Fallback to trust-based for non-GNN nodes
            best_node = None
            best_trust = -1
            
            for node_name in available_nodes.keys():
                trust_score = src_node.trust_mat.get(node_name, 0.5)
                if trust_score > best_trust:
                    best_trust = trust_score
                    best_node = node_name
                    
            return best_node or list(available_nodes.keys())[0]


class TrustEvaluationMetrics:
    """Comprehensive evaluation metrics for trust systems."""
    
    def __init__(self):
        self.trust_history = {}
        self.ground_truth = {}
        self.predictions = {}
        self.timestamps = []
        
    def record_trust_scores(self, timestamp: int, trust_scores: Dict[str, float], 
                          ground_truth: Dict[str, bool]):
        """Record trust scores and ground truth at a timestamp."""
        self.timestamps.append(timestamp)
        
        if timestamp not in self.trust_history:
            self.trust_history[timestamp] = {}
            self.ground_truth[timestamp] = {}
            
        self.trust_history[timestamp].update(trust_scores)
        self.ground_truth[timestamp].update(ground_truth)
        
    def compute_classification_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """Compute classification metrics."""
        all_predictions = []
        all_ground_truth = []
        all_trust_scores = []
        
        for timestamp in self.timestamps:
            for node_name in self.trust_history[timestamp]:
                trust_score = self.trust_history[timestamp][node_name]
                is_trustworthy = self.ground_truth[timestamp].get(node_name, True)
                
                all_trust_scores.append(trust_score)
                all_predictions.append(1 if trust_score >= threshold else 0)
                all_ground_truth.append(1 if is_trustworthy else 0)
        
        if not all_predictions:
            return {}
            
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(all_ground_truth, all_predictions)
        metrics['f1_macro'] = f1_score(all_ground_truth, all_predictions, average='macro', zero_division=0)
        metrics['f1_binary'] = f1_score(all_ground_truth, all_predictions, zero_division=0)
        
        # Handle MCC edge cases
        try:
            metrics['mcc'] = matthews_corrcoef(all_ground_truth, all_predictions)
        except:
            metrics['mcc'] = 0.0
            
        # Handle AUC edge cases  
        try:
            if len(set(all_ground_truth)) > 1:
                metrics['auc'] = roc_auc_score(all_ground_truth, all_trust_scores)
            else:
                metrics['auc'] = 0.5  # Random performance when only one class
        except:
            metrics['auc'] = 0.5
            
        # Regression metrics
        metrics['mae'] = mean_absolute_error(all_ground_truth, all_trust_scores)
        
        return metrics
    
    def compute_robustness_metrics(self, attack_periods: List[Tuple[int, int]]) -> Dict[str, float]:
        """Compute robustness metrics during attack periods."""
        attack_metrics = []
        normal_metrics = []
        
        for timestamp in self.timestamps:
            is_attack_period = any(start <= timestamp <= end for start, end in attack_periods)
            
            timestamp_scores = []
            timestamp_ground_truth = []
            
            for node_name in self.trust_history[timestamp]:
                trust_score = self.trust_history[timestamp][node_name]
                is_trustworthy = self.ground_truth[timestamp].get(node_name, True)
                
                timestamp_scores.append(trust_score)
                timestamp_ground_truth.append(1 if is_trustworthy else 0)
            
            if timestamp_scores:
                mae = mean_absolute_error(timestamp_ground_truth, timestamp_scores)
                
                if is_attack_period:
                    attack_metrics.append(mae)
                else:
                    normal_metrics.append(mae)
        
        return {
            'normal_mae': np.mean(normal_metrics) if normal_metrics else 0,
            'attack_mae': np.mean(attack_metrics) if attack_metrics else 0,
            'robustness_ratio': (np.mean(normal_metrics) / max(np.mean(attack_metrics), 0.001) 
                               if normal_metrics and attack_metrics else 1.0)
        }
    
    def plot_trust_evolution(self, output_dir: str, node_categories: Dict[str, str]):
        """Plot trust score evolution over time."""
        plt.figure(figsize=(15, 10))
        
        # Group nodes by category
        categories = {}
        for node_name, category in node_categories.items():
            if category not in categories:
                categories[category] = []
            categories[category].append(node_name)
        
        colors = {'honest': 'green', 'malicious': 'red', 'gnn': 'blue'}
        
        for category, nodes in categories.items():
            for node_name in nodes:
                trust_values = []
                timestamps = []
                
                for timestamp in sorted(self.timestamps):
                    if node_name in self.trust_history[timestamp]:
                        trust_values.append(self.trust_history[timestamp][node_name])
                        timestamps.append(timestamp)
                
                if trust_values:
                    plt.plot(timestamps, trust_values, 
                           color=colors.get(category, 'gray'), 
                           alpha=0.7, 
                           label=f'{category.title()} ({node_name})' if nodes.index(node_name) == 0 else "")
        
        plt.xlabel('Time Step')
        plt.ylabel('Trust Score')
        plt.title('Trust Score Evolution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trust_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_attack_impact(self, output_dir: str, attack_periods: List[Tuple[int, int]], 
                          attack_types: List[str]):
        """Plot the impact of attacks on trust scores."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Aggregate trust scores by timestamp
        avg_trust_by_time = {}
        for timestamp in sorted(self.timestamps):
            scores = list(self.trust_history[timestamp].values())
            avg_trust_by_time[timestamp] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        timestamps = sorted(avg_trust_by_time.keys())
        
        # Plot 1: Average trust over time with attack periods
        ax1 = axes[0, 0]
        means = [avg_trust_by_time[t]['mean'] for t in timestamps]
        stds = [avg_trust_by_time[t]['std'] for t in timestamps]
        
        ax1.plot(timestamps, means, 'b-', label='Average Trust')
        ax1.fill_between(timestamps, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)], 
                        alpha=0.3, color='blue')
        
        # Highlight attack periods
        for i, (start, end) in enumerate(attack_periods):
            ax1.axvspan(start, end, alpha=0.3, color='red', 
                       label=f'{attack_types[i] if i < len(attack_types) else "Attack"}' if i == 0 else "")
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Average Trust Score')
        ax1.set_title('Average Trust Score with Attack Periods')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trust variance over time
        ax2 = axes[0, 1]
        variances = [avg_trust_by_time[t]['std']**2 for t in timestamps]
        ax2.plot(timestamps, variances, 'g-', label='Trust Variance')
        
        for start, end in attack_periods:
            ax2.axvspan(start, end, alpha=0.3, color='red')
            
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Trust Variance')
        ax2.set_title('Trust Variance Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Min/Max trust scores
        ax3 = axes[1, 0]
        mins = [avg_trust_by_time[t]['min'] for t in timestamps]
        maxs = [avg_trust_by_time[t]['max'] for t in timestamps]
        
        ax3.plot(timestamps, mins, 'r-', label='Minimum Trust')
        ax3.plot(timestamps, maxs, 'g-', label='Maximum Trust')
        
        for start, end in attack_periods:
            ax3.axvspan(start, end, alpha=0.3, color='red')
            
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Trust Score')  
        ax3.set_title('Min/Max Trust Scores')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Attack impact metrics
        ax4 = axes[1, 1]
        
        # Calculate metrics for each time window
        window_size = max(1, len(timestamps) // 20)
        windowed_metrics = []
        window_times = []
        
        for i in range(0, len(timestamps), window_size):
            window_end = min(i + window_size, len(timestamps))
            window_timestamps = timestamps[i:window_end]
            
            if window_timestamps:
                window_means = [avg_trust_by_time[t]['mean'] for t in window_timestamps]
                windowed_metrics.append(np.mean(window_means))
                window_times.append(np.mean(window_timestamps))
        
        if windowed_metrics:
            ax4.plot(window_times, windowed_metrics, 'purple', marker='o', label='Windowed Average')
            
            for start, end in attack_periods:
                ax4.axvspan(start, end, alpha=0.3, color='red')
                
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Windowed Average Trust')
        ax4.set_title('Windowed Trust Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/attack_impact.png', dpi=300, bbox_inches='tight')
        plt.close()


class ComprehensiveAttackExperiment:
    """Main experiment framework for comprehensive attack testing."""
    
    def __init__(self, output_dir: str = "experiments/attack_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        self.results = []
        self.trust_metrics = TrustEvaluationMetrics()
        
    def create_attack_scenario_config(self, base_config_path: str, 
                                    num_malicious: int = 3) -> str:
        """Create scenario configuration with malicious nodes."""
        with open(base_config_path, 'r') as f:
            config = json.load(f)
        
        # Convert some nodes to malicious
        node_count = 0
        for node in config['Nodes']:
            if node_count < num_malicious:
                node['NodeType'] = 'MaliciousNode'
            elif node_count < num_malicious + 3:
                node['NodeType'] = 'GNNTrustNode'
            else:
                node['NodeType'] = 'TrustNode'
            node_count += 1
            
        # Save attack scenario config
        attack_config_path = f"{self.output_dir}/attack_scenario_config.json"
        with open(attack_config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return attack_config_path
        
    def run_attack_experiment(self, attack_types: List[str], 
                            allocation_strategies: List[str],
                            num_tasks: int = 100,
                            simulation_time: int = 1000) -> Dict[str, Any]:
        """Run comprehensive attack experiment."""
        
        print(f"\n=== Attack Experiment: {', '.join(attack_types)} ===")
        print(f"Allocation strategies: {', '.join(allocation_strategies)}")
        
        results = {}
        
        for strategy in allocation_strategies:
            print(f"\nTesting allocation strategy: {strategy}")
            
            # Create scenario with malicious nodes
            base_config = "eval/benchmarks/Topo4MEC/data/25N50E/config.json"
            attack_config_path = self.create_attack_scenario_config(base_config, num_malicious=5)
            
            scenario = Scenario(config_file=attack_config_path)
            env_config_path = self.create_env_config(len(scenario.get_nodes()))
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
            
            print(f"Malicious nodes: {malicious_nodes}")
            print(f"GNN nodes: {gnn_nodes}")
            print(f"Honest nodes: {honest_nodes}")
            
            # Track results
            strategy_results = {
                'successful_tasks': 0,
                'failed_tasks': 0,
                'tasks_to_malicious': 0,
                'tasks_to_honest': 0,
                'trust_evolution': {},
                'attack_periods': []
            }
            
            # Run simulation with tasks and attacks
            task_counter = 0
            attack_periods = []
            
            for time_step in range(0, simulation_time, 10):
                attack_simulator.time_step = time_step
                
                # Simulate attacks based on attack types
                if time_step % 100 == 0 and time_step > 0:  # Attack every 100 time steps
                    attack_start = time_step
                    attack_end = time_step + 50
                    attack_periods.append((attack_start, attack_end))
                    
                    for attack_type in attack_types:
                        if attack_type == 'on_off':
                            attack_simulator.simulate_on_off_attack(
                                malicious_nodes, [20, 30], [30, 40])
                        elif attack_type == 'ballot_stuffing':
                            attack_simulator.simulate_ballot_stuffing_attack(
                                malicious_nodes, honest_nodes + gnn_nodes, 0.9)
                        elif attack_type == 'bad_mouthing':
                            attack_simulator.simulate_bad_mouthing_attack(
                                malicious_nodes, honest_nodes + gnn_nodes, 0.1)
                        elif attack_type == 'collusion':
                            if len(malicious_nodes) >= 2:
                                attack_simulator.simulate_collusion_attack(
                                    malicious_nodes[:2], honest_nodes + gnn_nodes)
                        elif attack_type == 'sybil':
                            if len(malicious_nodes) >= 3:
                                attack_simulator.simulate_sybil_attack(
                                    malicious_nodes[1:3], malicious_nodes[0])
                
                # Generate and process tasks
                if time_step % 10 == 0 and task_counter < num_tasks:
                    # Create task
                    task = Task(
                        task_id=task_counter + 1,
                        task_size=100 + np.random.randint(50, 200),
                        cycles_per_bit=5,
                        trans_bit_rate=50,
                        src_name=np.random.choice(gnn_nodes + honest_nodes),
                        ddl=time_step + 100 + np.random.randint(50, 150),
                        task_name=f'attack_task_{task_counter + 1}'
                    )
                    
                    # Select destination based on strategy
                    src_node = scenario.get_node(task.src_name)
                    available_nodes = {name: node for name, node in scenario.get_nodes().items() 
                                     if name != task.src_name}
                    
                    if strategy == 'round_robin':
                        dst_name = TaskAllocationStrategy.round_robin(
                            list(available_nodes.keys()), task_counter)
                    elif strategy == 'random':
                        dst_name = TaskAllocationStrategy.random_allocation(
                            list(available_nodes.keys()))
                    elif strategy == 'greedy':
                        dst_name = TaskAllocationStrategy.greedy_resources(available_nodes)
                    elif strategy == 'gnn_trust':
                        dst_name = TaskAllocationStrategy.gnn_trust_based(
                            src_node, available_nodes, task)
                    else:
                        dst_name = list(available_nodes.keys())[0]
                    
                    # Process task
                    try:
                        env.process(task=task, dst_name=dst_name)
                        strategy_results['successful_tasks'] += 1
                        
                        if dst_name in malicious_nodes:
                            strategy_results['tasks_to_malicious'] += 1
                        else:
                            strategy_results['tasks_to_honest'] += 1
                            
                    except Exception as e:
                        strategy_results['failed_tasks'] += 1
                        print(f"Task {task_counter + 1} failed: {e}")
                    
                    task_counter += 1
                
                # Record trust scores every 20 time steps
                if time_step % 20 == 0:
                    trust_scores = {}
                    ground_truth = {}
                    
                    for name, node in scenario.get_nodes().items():
                        if isinstance(node, GNNTrustNode):
                            # Get trust scores from GNN
                            for target_name in scenario.get_nodes().keys():
                                if target_name != name:
                                    trust_key = f"{name}->{target_name}"
                                    trust_scores[trust_key] = node.compute_trust(target_name)
                                    ground_truth[trust_key] = target_name not in malicious_nodes
                        elif isinstance(node, TrustNode):
                            # Get trust scores from traditional trust
                            for target_name, trust_value in node.trust_mat.items():
                                if target_name != name:
                                    trust_key = f"{name}->{target_name}"
                                    trust_scores[trust_key] = trust_value
                                    ground_truth[trust_key] = target_name not in malicious_nodes
                    
                    self.trust_metrics.record_trust_scores(time_step, trust_scores, ground_truth)
                    strategy_results['trust_evolution'][time_step] = trust_scores.copy()
                
                # Advance attack simulator
                attack_simulator.advance_time()
            
            # Run remaining simulation
            env.run(simulation_time)
            
            # Calculate final metrics
            strategy_results['total_tasks'] = strategy_results['successful_tasks'] + strategy_results['failed_tasks']
            strategy_results['success_rate'] = (strategy_results['successful_tasks'] / 
                                              max(1, strategy_results['total_tasks']))
            strategy_results['malicious_task_ratio'] = (strategy_results['tasks_to_malicious'] / 
                                                      max(1, strategy_results['successful_tasks']))
            strategy_results['attack_periods'] = attack_periods
            
            results[strategy] = strategy_results
            
            print(f"Results for {strategy}:")
            print(f"  Success rate: {strategy_results['success_rate']:.2%}")
            print(f"  Tasks to malicious nodes: {strategy_results['malicious_task_ratio']:.2%}")
        
        return results
    
    def create_env_config(self, num_nodes: int) -> str:
        """Create environment config."""
        env_config = {
            "Basic": {
                "VisFrame": "off",
                "Train": "off", 
                "Test": "off"
            },
            "VisFrame": {
                "LogInfoPath": f"{self.output_dir}/logs/vis",
                "LogFramesPath": f"{self.output_dir}/logs/vis/frames",
                "TargetNodeList": []
            },
            "Train": {
                "CloseLogger": "True"
            },
            "Test": {}
        }
        
        env_config_path = f"{self.output_dir}/attack_env_config.json"
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f, indent=2)
            
        return env_config_path
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive attack evaluation with all scenarios."""
        print("=== Comprehensive Attack Evaluation ===")
        
        # Define experiment scenarios
        individual_attacks = [
            ['on_off'],
            ['ballot_stuffing'], 
            ['bad_mouthing'],
            ['collusion'],
            ['sybil']
        ]
        
        combined_attacks = [
            ['on_off', 'ballot_stuffing'],
            ['bad_mouthing', 'collusion'],
            ['on_off', 'sybil'],
            ['ballot_stuffing', 'bad_mouthing', 'collusion']
        ]
        
        allocation_strategies = ['round_robin', 'random', 'greedy', 'gnn_trust']
        
        all_results = {}
        
        # Individual attack scenarios
        print("\n--- Individual Attack Scenarios ---")
        for attacks in individual_attacks:
            attack_name = '_'.join(attacks)
            print(f"\nRunning {attack_name} attacks...")
            
            results = self.run_attack_experiment(
                attack_types=attacks,
                allocation_strategies=allocation_strategies,
                num_tasks=50,
                simulation_time=500
            )
            
            all_results[f"individual_{attack_name}"] = results
        
        # Combined attack scenarios  
        print("\n--- Combined Attack Scenarios ---")
        for attacks in combined_attacks:
            attack_name = '_'.join(attacks)
            print(f"\nRunning combined {attack_name} attacks...")
            
            results = self.run_attack_experiment(
                attack_types=attacks,
                allocation_strategies=allocation_strategies,
                num_tasks=75,
                simulation_time=750
            )
            
            all_results[f"combined_{attack_name}"] = results
        
        # Compute comprehensive metrics
        self.compute_and_save_metrics(all_results)
        self.generate_visualizations(all_results)
        
        return all_results
    
    def compute_and_save_metrics(self, all_results: Dict[str, Any]):
        """Compute and save comprehensive evaluation metrics."""
        print("\n--- Computing Evaluation Metrics ---")
        
        # Classification metrics
        classification_metrics = self.trust_metrics.compute_classification_metrics()
        print("Classification Metrics:")
        for metric, value in classification_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Extract attack periods from results
        all_attack_periods = []
        for scenario_name, scenario_results in all_results.items():
            for strategy, strategy_results in scenario_results.items():
                if 'attack_periods' in strategy_results:
                    all_attack_periods.extend(strategy_results['attack_periods'])
        
        # Remove duplicates and sort
        all_attack_periods = sorted(list(set(all_attack_periods)))
        
        # Robustness metrics
        robustness_metrics = self.trust_metrics.compute_robustness_metrics(all_attack_periods)
        print("\nRobustness Metrics:")
        for metric, value in robustness_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Strategy comparison
        strategy_comparison = {}
        for scenario_name, scenario_results in all_results.items():
            strategy_comparison[scenario_name] = {}
            for strategy, results in scenario_results.items():
                strategy_comparison[scenario_name][strategy] = {
                    'success_rate': results['success_rate'],
                    'malicious_task_ratio': results['malicious_task_ratio'],
                    'total_tasks': results['total_tasks']
                }
        
        # Save all metrics
        comprehensive_metrics = {
            'classification_metrics': classification_metrics,
            'robustness_metrics': robustness_metrics,
            'strategy_comparison': strategy_comparison,
            'timestamp': time.time()
        }
        
        with open(f"{self.output_dir}/comprehensive_metrics.json", 'w') as f:
            json.dump(comprehensive_metrics, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for scenario_name, scenario_results in all_results.items():
            for strategy, results in scenario_results.items():
                summary_data.append({
                    'Scenario': scenario_name,
                    'Strategy': strategy,
                    'Success_Rate': f"{results['success_rate']:.2%}",
                    'Malicious_Task_Ratio': f"{results['malicious_task_ratio']:.2%}",
                    'Total_Tasks': results['total_tasks']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_dir}/attack_experiment_summary.csv", index=False)
        
        print(f"\nMetrics saved to {self.output_dir}/")
    
    def generate_visualizations(self, all_results: Dict[str, Any]):
        """Generate comprehensive visualizations."""
        print("\n--- Generating Visualizations ---")
        
        # Determine node categories for plotting
        node_categories = {
            'n0': 'malicious', 'n1': 'malicious', 'n2': 'malicious', 'n3': 'malicious', 'n4': 'malicious',
            'n5': 'gnn', 'n6': 'gnn', 'n7': 'gnn',
            'n8': 'honest', 'n9': 'honest', 'n10': 'honest'
        }
        
        # Plot trust evolution
        self.trust_metrics.plot_trust_evolution(f"{self.output_dir}/plots", node_categories)
        
        # Extract attack periods for plotting
        all_attack_periods = []
        attack_types = []
        for scenario_name, scenario_results in all_results.items():
            for strategy, strategy_results in scenario_results.items():
                if 'attack_periods' in strategy_results:
                    all_attack_periods.extend(strategy_results['attack_periods'])
                    attack_types.extend([scenario_name.split('_')[1]] * len(strategy_results['attack_periods']))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_attack_periods = []
        unique_attack_types = []
        for period, attack_type in zip(all_attack_periods, attack_types):
            if period not in seen:
                unique_attack_periods.append(period)
                unique_attack_types.append(attack_type)
                seen.add(period)
        
        # Plot attack impact
        self.trust_metrics.plot_attack_impact(f"{self.output_dir}/plots", 
                                            unique_attack_periods, unique_attack_types)
        
        # Strategy comparison visualization
        self.plot_strategy_comparison(all_results)
        
        print(f"Visualizations saved to {self.output_dir}/plots/")
    
    def plot_strategy_comparison(self, all_results: Dict[str, Any]):
        """Plot strategy comparison across different attack scenarios."""
        # Prepare data for visualization
        scenarios = []
        strategies = []
        success_rates = []
        malicious_ratios = []
        
        for scenario_name, scenario_results in all_results.items():
            for strategy, results in scenario_results.items():
                scenarios.append(scenario_name.replace('_', ' ').title())
                strategies.append(strategy.replace('_', ' ').title())
                success_rates.append(results['success_rate'])
                malicious_ratios.append(results['malicious_task_ratio'])
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: Success Rate by Strategy
        df_success = pd.DataFrame({
            'Scenario': scenarios,
            'Strategy': strategies,
            'Success_Rate': success_rates
        })
        
        pivot_success = df_success.pivot(index='Scenario', columns='Strategy', values='Success_Rate')
        sns.heatmap(pivot_success, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[0,0])
        axes[0,0].set_title('Success Rate by Strategy and Scenario')
        axes[0,0].set_xlabel('Allocation Strategy')
        axes[0,0].set_ylabel('Attack Scenario')
        
        # Plot 2: Malicious Task Ratio by Strategy
        df_malicious = pd.DataFrame({
            'Scenario': scenarios,
            'Strategy': strategies,
            'Malicious_Ratio': malicious_ratios
        })
        
        pivot_malicious = df_malicious.pivot(index='Scenario', columns='Strategy', values='Malicious_Ratio')
        sns.heatmap(pivot_malicious, annot=True, fmt='.2%', cmap='RdYlBu_r', ax=axes[0,1])
        axes[0,1].set_title('Malicious Task Ratio by Strategy and Scenario')
        axes[0,1].set_xlabel('Allocation Strategy')
        axes[0,1].set_ylabel('Attack Scenario')
        
        # Plot 3: Strategy Performance Comparison
        strategy_avg = df_success.groupby('Strategy')['Success_Rate'].mean().sort_values(ascending=False)
        axes[1,0].bar(range(len(strategy_avg)), strategy_avg.values, 
                     color=['blue', 'green', 'orange', 'red'])
        axes[1,0].set_xticks(range(len(strategy_avg)))
        axes[1,0].set_xticklabels(strategy_avg.index, rotation=45)
        axes[1,0].set_ylabel('Average Success Rate')
        axes[1,0].set_title('Average Success Rate by Strategy')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Attack Impact Comparison
        scenario_impact = df_malicious.groupby('Scenario')['Malicious_Ratio'].mean().sort_values(ascending=False)
        axes[1,1].bar(range(len(scenario_impact)), scenario_impact.values, 
                     color='red', alpha=0.7)
        axes[1,1].set_xticks(range(len(scenario_impact)))
        axes[1,1].set_xticklabels(scenario_impact.index, rotation=45, ha='right')
        axes[1,1].set_ylabel('Average Malicious Task Ratio')
        axes[1,1].set_title('Attack Impact by Scenario')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run comprehensive attack experiments."""
    print("=== Comprehensive Attack Simulation and Evaluation ===")
    
    # Create experiment framework
    experiment = ComprehensiveAttackExperiment()
    
    # Run comprehensive evaluation
    results = experiment.run_comprehensive_evaluation()
    
    print("\n=== Experiment Complete ===")
    print(f"Results saved to: {experiment.output_dir}")
    print(f"Visualizations saved to: {experiment.output_dir}/plots")


if __name__ == '__main__':
    main()