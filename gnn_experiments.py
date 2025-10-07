#!/usr/bin/env python3
"""
Comprehensive GNN Trust Experiments Runner
Tests GNN-based trust system with different datasets and configurations.
"""

import sys
import os
import json
import time
import traceback
from typing import Dict, List, Any
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

from core.env import Env_Trust
from core.task import Task
from examples.scenarios.gnn_scenario import Scenario as GNNScenario
from eval.benchmarks.Topo4MEC.scenario import Scenario as Topo4MECScenario
from eval.benchmarks.Pakistan.scenario import Scenario as PakistanScenario
from zoo.gnn_node import GNNTrustNode


class GNNTrustExperiment:
    """Experimental framework for GNN Trust evaluation."""
    
    def __init__(self, output_dir: str = "experiments/gnn_trust"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
    def create_gnn_config_for_dataset(self, dataset_name: str, dataset_flag: str, 
                                    num_nodes: int) -> str:
        """Create a GNN-specific config file for a dataset."""
        if dataset_name == "topo4mec":
            base_config_path = f"eval/benchmarks/Topo4MEC/data/{dataset_flag}/config.json"
        elif dataset_name == "pakistan":
            base_config_path = f"eval/benchmarks/Pakistan/data/{dataset_flag}/config.json"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        # Load base config
        with open(base_config_path, 'r') as f:
            config = json.load(f)
        
        # Convert some nodes to GNNTrustNodes and some to MaliciousNodes
        gnn_nodes = min(5, num_nodes // 2)  # About half as GNN nodes
        malicious_nodes = min(2, num_nodes // 5)  # About 20% as malicious
        
        node_count = 0
        for node in config['Nodes']:
            if node_count < gnn_nodes:
                node['NodeType'] = 'GNNTrustNode'
            elif node_count < gnn_nodes + malicious_nodes:
                node['NodeType'] = 'MaliciousNode'
            else:
                node['NodeType'] = 'TrustNode'
            node_count += 1
            
        # Save GNN-specific config
        gnn_config_path = f"{self.output_dir}/gnn_{dataset_name}_{dataset_flag}_config.json"
        with open(gnn_config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return gnn_config_path
    
    def create_env_config_for_dataset(self, num_nodes: int, scenario) -> str:
        """Create environment config for a dataset."""
        # Get actual node names from scenario
        actual_node_names = list(scenario.get_nodes().keys())
        
        # Limit to 20 nodes for visualization
        target_nodes = actual_node_names[:min(20, len(actual_node_names))]
        
        env_config = {
            "Basic": {
                "VisFrame": "off",  # Disabled for large datasets
                "Train": "off",
                "Test": "off"
            },
            "VisFrame": {
                "LogInfoPath": f"{self.output_dir}/logs/vis",
                "LogFramesPath": f"{self.output_dir}/logs/vis/frames",
                "TargetNodeList": target_nodes
            },
            "Train": {
                "CloseLogger": "True"
            },
            "Test": {}
        }
        
        env_config_path = f"{self.output_dir}/env_config_{num_nodes}nodes.json"
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f, indent=2)
            
        return env_config_path
    
    def run_experiment(self, dataset_name: str, dataset_flag: str, 
                      num_tasks: int = 10, simulation_time: int = 500) -> Dict[str, Any]:
        """Run a single experiment."""
        print(f"\n=== Running Experiment: {dataset_name.upper()} - {dataset_flag} ===")
        
        experiment_start_time = time.time()
        result = {
            'dataset': dataset_name,
            'dataset_flag': dataset_flag,
            'num_tasks': num_tasks,
            'simulation_time': simulation_time,
            'start_time': experiment_start_time,
            'status': 'failed',
            'error': None,
            'metrics': {}
        }
        
        try:
            # Create scenario based on dataset type
            if dataset_name == "topo4mec":
                config_path = f"eval/benchmarks/Topo4MEC/data/{dataset_flag}/config.json"
                gnn_config_path = self.create_gnn_config_for_dataset(dataset_name, dataset_flag, 25)
                scenario = GNNScenario(config_file=gnn_config_path)
            elif dataset_name == "pakistan":
                config_path = f"eval/benchmarks/Pakistan/data/{dataset_flag}/config.json"
                gnn_config_path = self.create_gnn_config_for_dataset(dataset_name, dataset_flag, 8)
                scenario = GNNScenario(config_file=gnn_config_path)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            
            # Create environment config
            num_nodes = len(scenario.get_nodes())
            env_config_path = self.create_env_config_for_dataset(num_nodes, scenario)
            
            # Create environment and set infrastructure for GNN nodes
            env = Env_Trust(scenario, config_file=env_config_path)
            
            print(f"Scenario loaded with {num_nodes} nodes")
            
            # Set infrastructure for all GNNTrustNodes
            gnn_node_count = 0
            for name, node in env.scenario.get_nodes().items():
                if isinstance(node, GNNTrustNode):
                    node.set_infrastructure(env.scenario.infrastructure)
                    gnn_node_count += 1
            
            print(f"Configured {gnn_node_count} GNN nodes")
            
            # Get available nodes for task generation
            all_nodes = list(env.scenario.get_nodes().keys())
            source_nodes = all_nodes[:min(3, len(all_nodes))]  # Use first few nodes as sources
            
            # Run multiple tasks
            successful_selections = 0
            failed_selections = 0
            trust_scores = []
            selection_times = []
            
            for task_id in range(1, num_tasks + 1):
                try:
                    # Select random source node
                    src_name = source_nodes[(task_id - 1) % len(source_nodes)]
                    src_node = env.scenario.get_node(src_name)
                    
                    # Create task
                    task = Task(
                        task_id=task_id,
                        task_size=100 + (task_id * 50) % 500,  # Variable task sizes
                        cycles_per_bit=5,
                        trans_bit_rate=50,
                        src_name=src_name,
                        ddl=200 + (task_id * 20) % 300,  # Variable deadlines
                        task_name=f'exp_task_{task_id}'
                    )
                    
                    # Get available destination nodes
                    available_nodes = {name: node for name, node in env.scenario.get_nodes().items() 
                                     if name != src_name}
                    
                    # Test node selection
                    selection_start = time.time()
                    if isinstance(src_node, GNNTrustNode):
                        dst_name = src_node.select_node(task, available_nodes)
                        if dst_name:
                            trust_score = src_node.compute_trust(dst_name)
                            trust_scores.append(trust_score)
                            successful_selections += 1
                        else:
                            failed_selections += 1
                            dst_name = list(available_nodes.keys())[0]  # Fallback
                    else:
                        # Non-GNN node, use simple selection
                        dst_name = list(available_nodes.keys())[0]
                        trust_scores.append(0.5)  # Default trust
                        successful_selections += 1
                    
                    selection_time = time.time() - selection_start
                    selection_times.append(selection_time)
                    
                    # Process the task
                    env.process(task=task, dst_name=dst_name)
                    
                    if task_id % 5 == 0:
                        print(f"Processed task {task_id}/{num_tasks}")
                        
                except Exception as e:
                    print(f"Error in task {task_id}: {e}")
                    failed_selections += 1
            
            # Run simulation
            print("Running simulation...")
            sim_start = time.time()
            env.run(simulation_time)
            sim_time = time.time() - sim_start
            
            # Collect metrics
            result['metrics'] = {
                'num_nodes': num_nodes,
                'gnn_nodes': gnn_node_count,
                'successful_selections': successful_selections,
                'failed_selections': failed_selections,
                'selection_success_rate': successful_selections / (successful_selections + failed_selections) if (successful_selections + failed_selections) > 0 else 0,
                'avg_trust_score': sum(trust_scores) / len(trust_scores) if trust_scores else 0,
                'min_trust_score': min(trust_scores) if trust_scores else 0,
                'max_trust_score': max(trust_scores) if trust_scores else 0,
                'avg_selection_time': sum(selection_times) / len(selection_times) if selection_times else 0,
                'max_selection_time': max(selection_times) if selection_times else 0,
                'total_processed_tasks': env.task_count,
                'completed_tasks': len(env.done_task_info),
                'simulation_time': sim_time,
                'total_experiment_time': time.time() - experiment_start_time
            }
            
            result['status'] = 'success'
            print(f"Experiment completed successfully!")
            print(f"Selection success rate: {result['metrics']['selection_success_rate']:.2%}")
            print(f"Average trust score: {result['metrics']['avg_trust_score']:.4f}")
            print(f"Total processed tasks: {result['metrics']['total_processed_tasks']}")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            print(f"Experiment failed: {e}")
            print("Full traceback:")
            print(result['traceback'])
        
        result['end_time'] = time.time()
        result['duration'] = result['end_time'] - result['start_time']
        
        return result
    
    def run_all_experiments(self):
        """Run experiments on all available datasets."""
        print("=== GNN Trust System Comprehensive Experiments ===")
        
        # Experiment configurations
        experiments = [
            # Topo4MEC datasets
            ('topo4mec', '25N50E', 5, 300),
            ('topo4mec', '50N50E', 8, 400),
            ('topo4mec', '100N150E', 10, 500),
            
            # Pakistan datasets  
            ('pakistan', 'Tuple30K', 5, 300),
            ('pakistan', 'Tuple50K', 8, 400),
            ('pakistan', 'Tuple100K', 10, 500),
        ]
        
        for dataset_name, dataset_flag, num_tasks, sim_time in experiments:
            result = self.run_experiment(dataset_name, dataset_flag, num_tasks, sim_time)
            self.results.append(result)
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save experiment results to files."""
        # Save detailed results as JSON
        results_file = f"{self.output_dir}/experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary as CSV
        summary_data = []
        for result in self.results:
            if result['status'] == 'success':
                summary_data.append({
                    'Dataset': f"{result['dataset']}-{result['dataset_flag']}",
                    'Nodes': result['metrics']['num_nodes'],
                    'GNN_Nodes': result['metrics']['gnn_nodes'],
                    'Tasks': result['num_tasks'],
                    'Success_Rate': f"{result['metrics']['selection_success_rate']:.2%}",
                    'Avg_Trust': f"{result['metrics']['avg_trust_score']:.4f}",
                    'Processed_Tasks': result['metrics']['total_processed_tasks'],
                    'Completed_Tasks': result['metrics']['completed_tasks'],
                    'Avg_Selection_Time': f"{result['metrics']['avg_selection_time']:.4f}s",
                    'Total_Time': f"{result['duration']:.2f}s",
                    'Status': result['status']
                })
            else:
                summary_data.append({
                    'Dataset': f"{result['dataset']}-{result['dataset_flag']}",
                    'Status': result['status'],
                    'Error': result.get('error', 'Unknown')
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"{self.output_dir}/experiment_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  Detailed: {results_file}")
        print(f"  Summary: {summary_file}")
    
    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        successful_experiments = [r for r in self.results if r['status'] == 'success']
        failed_experiments = [r for r in self.results if r['status'] == 'failed']
        
        print(f"Total experiments: {len(self.results)}")
        print(f"Successful: {len(successful_experiments)}")
        print(f"Failed: {len(failed_experiments)}")
        
        if successful_experiments:
            print("\n--- Successful Experiments ---")
            for result in successful_experiments:
                metrics = result['metrics']
                print(f"{result['dataset']}-{result['dataset_flag']}:")
                print(f"  Nodes: {metrics['num_nodes']} (GNN: {metrics['gnn_nodes']})")
                print(f"  Selection Success: {metrics['selection_success_rate']:.2%}")
                print(f"  Avg Trust Score: {metrics['avg_trust_score']:.4f}")
                print(f"  Tasks: {metrics['total_processed_tasks']} processed, {metrics['completed_tasks']} completed")
                print(f"  Time: {result['duration']:.2f}s")
                print()
        
        if failed_experiments:
            print("\n--- Failed Experiments ---")
            for result in failed_experiments:
                print(f"{result['dataset']}-{result['dataset_flag']}: {result.get('error', 'Unknown error')}")


def main():
    """Main experiment runner."""
    print("Starting GNN Trust System Experiments...")
    
    # Create experiment runner
    experiment = GNNTrustExperiment()
    
    # Run all experiments
    experiment.run_all_experiments()
    
    print("\nAll experiments completed!")


if __name__ == '__main__':
    main()