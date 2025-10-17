"""
Comprehensive GNN Trust Training and Evaluation Script.
This script trains multiple GNN models for trust prediction and compares their performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import networkx as nx
import torch
import json
import time
from collections import defaultdict

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.env import Env_Trust
from core.task import Task
from examples.scenarios.trust_scenario_1 import Scenario

from policies.gnn_trust.training import TrustTrainer
from policies.gnn_trust.feature_engineering_simple import extract_node_features
from policies.gnn_trust.models import GATModel, GraphSAGEModel, GCNModel, TrustGraphTransformer

def create_synthetic_trust_data(env, num_samples=1000):
    """Create synthetic trust data for training.
    
    Args:
        env: Environment instance
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (graphs, trust_labels)
    """
    graphs = []
    trust_labels = []
    
    print(f"Generating {num_samples} synthetic trust samples...")
    
    # Get base graph structure
    base_graph = env.scenario.infrastructure.graph.copy()
    nodes = list(env.scenario.get_nodes().items())
    
    # Generate samples with different trust patterns
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} samples")
        
        # Create graph copy
        graph = base_graph.copy()
        
        # Add node data
        for node_name, node in nodes:
            graph.nodes[node_name]['data'] = node
        
        # Simulate different trust scenarios
        trust_dict = {}
        
        for node_name, node in nodes:
            # Base trust score
            base_trust = 0.7
            
            # Add some randomness
            noise = np.random.normal(0, 0.1)
            
            # Simulate malicious behavior patterns
            if hasattr(node, 'is_malicious') and node.is_malicious:
                # Malicious nodes have lower trust
                trust_score = max(0.0, base_trust - 0.4 + noise)
            else:
                # Good nodes have higher trust with some variation
                trust_score = min(1.0, base_trust + 0.2 + noise)
            
            # Simulate performance-based trust
            if hasattr(node, 'get_online') and not node.get_online():
                trust_score *= 0.3  # Offline nodes have very low trust
            
            # Add CPU and buffer utilization effects
            if hasattr(node, 'free_cpu_freq') and hasattr(node, 'max_cpu_freq'):
                cpu_util = 1.0 - (node.free_cpu_freq / max(1.0, node.max_cpu_freq))
                if cpu_util > 0.9:  # Overloaded nodes
                    trust_score *= 0.8
                elif cpu_util < 0.3:  # Underutilized nodes (might be suspicious)
                    trust_score *= 0.9
            
            trust_dict[node_name] = float(trust_score)
        
        # Add trust scores to edges
        for src, dst in graph.edges():
            # Edge trust is influenced by destination node trust
            dst_trust = trust_dict.get(dst, 0.5)
            edge_noise = np.random.normal(0, 0.05)
            edge_trust = max(0.0, min(1.0, dst_trust + edge_noise))
            
            # Handle different NetworkX edge data formats
            if graph.has_edge(src, dst):
                if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
                    # For multigraphs, need to specify key
                    for key in graph[src][dst]:
                        graph[src][dst][key]['trust'] = edge_trust
                        break  # Just set the first edge
                else:
                    # Simple graph
                    graph[src][dst]['trust'] = edge_trust
        
        graphs.append(graph)
        trust_labels.append(trust_dict)
    
    print(f"Generated {len(graphs)} training samples")
    return graphs, trust_labels

def create_attack_scenarios(env, num_scenarios=100):
    """Create specific attack scenarios for evaluation.
    
    Args:
        env: Environment instance
        num_scenarios: Number of attack scenarios
        
    Returns:
        Tuple of (graphs, trust_labels, attack_info)
    """
    graphs = []
    trust_labels = []
    attack_info = []
    
    print(f"Generating {num_scenarios} attack scenarios...")
    
    base_graph = env.scenario.infrastructure.graph.copy()
    nodes = list(env.scenario.get_nodes().items())
    
    attack_types = ['sybil', 'ballot_stuffing', 'bad_mouthing', 'selective_service', 'collusion']
    
    for i in range(num_scenarios):
        # Create graph copy
        graph = base_graph.copy()
        
        # Add node data
        for node_name, node in nodes:
            graph.nodes[node_name]['data'] = node
        
        # Select attack type and parameters
        attack_type = np.random.choice(attack_types)
        num_attackers = np.random.randint(1, max(2, len(nodes) // 3))
        
        # Select attacker nodes
        attacker_nodes = np.random.choice([name for name, _ in nodes], num_attackers, replace=False)
        
        trust_dict = {}
        
        # Initialize base trust scores
        for node_name, node in nodes:
            if node_name in attacker_nodes:
                # Attackers have low trust but might try to hide it
                if attack_type == 'sybil':
                    # Sybil nodes might have artificially high initial trust
                    trust_score = np.random.uniform(0.6, 0.9)
                else:
                    trust_score = np.random.uniform(0.1, 0.4)
            else:
                # Honest nodes have good trust
                trust_score = np.random.uniform(0.6, 0.9)
            
            trust_dict[node_name] = float(trust_score)
        
        # Apply attack-specific modifications
        if attack_type == 'ballot_stuffing':
            # Attackers give very low ratings to honest nodes
            for attacker in attacker_nodes:
                for honest_node in trust_dict:
                    if honest_node not in attacker_nodes:
                        trust_dict[honest_node] *= 0.7  # Reduce trust due to bad ratings
        
        elif attack_type == 'bad_mouthing':
            # Similar to ballot stuffing but more extreme
            for attacker in attacker_nodes:
                for honest_node in trust_dict:
                    if honest_node not in attacker_nodes:
                        trust_dict[honest_node] *= 0.5
        
        elif attack_type == 'collusion':
            # Attackers boost each other's trust
            for attacker in attacker_nodes:
                trust_dict[attacker] = min(1.0, trust_dict[attacker] * 1.5)
        
        # Add trust scores to edges
        for src, dst in graph.edges():
            dst_trust = trust_dict.get(dst, 0.5)
            edge_noise = np.random.normal(0, 0.02)
            edge_trust = max(0.0, min(1.0, dst_trust + edge_noise))
            
            # Handle different NetworkX edge data formats
            if graph.has_edge(src, dst):
                if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
                    # For multigraphs, need to specify key
                    for key in graph[src][dst]:
                        graph[src][dst][key]['trust'] = edge_trust
                        break  # Just set the first edge
                else:
                    # Simple graph
                    graph[src][dst]['trust'] = edge_trust
        
        graphs.append(graph)
        trust_labels.append(trust_dict)
        attack_info.append({
            'attack_type': attack_type,
            'attackers': attacker_nodes.tolist(),
            'num_attackers': num_attackers
        })
    
    print(f"Generated {len(graphs)} attack scenarios")
    return graphs, trust_labels, attack_info

def train_and_evaluate_models(graphs, trust_labels, test_graphs=None, test_labels=None):
    """Train and evaluate multiple GNN models.
    
    Args:
        graphs: Training graphs
        trust_labels: Training trust labels
        test_graphs: Test graphs (optional)
        test_labels: Test trust labels (optional)
        
    Returns:
        Dictionary of trained models and results
    """
    models = ['gat', 'graphsage', 'gcn', 'transformer']
    results = {}
    
    # Enhanced training configuration
    config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 150,
        'early_stopping_patience': 25,
        'val_split': 0.2,
        'test_split': 0.1,
        'trust_threshold': 0.6,
        'pos_weight': 2.0,
        'gradient_clip': 1.0,
        'use_scheduler': True,
        'scheduler_patience': 10,
        'log_interval': 20
    }
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*60}")
        
        # Create trainer
        trainer = TrustTrainer(model_type=model_type, config=config)
        
        # Determine input dimension from a sample
        sample_graph = graphs[0]
        combined_features = extract_node_features(sample_graph)
        input_dim = combined_features.shape[1]
        
        print(f"Input dimension: {input_dim}")
        
        # Create model
        trainer.create_model(input_dim)
        
        # Prepare datasets
        train_loader, val_loader, test_loader = trainer.prepare_datasets(graphs, trust_labels)
        
        # Train model
        start_time = time.time()
        trainer.train(train_loader, val_loader)
        training_time = time.time() - start_time
        
        # Test model
        test_metrics = {}
        if test_loader:
            test_metrics = trainer.test(test_loader)
        elif test_graphs and test_labels:
            # Use external test set
            test_train_loader, _, ext_test_loader = trainer.prepare_datasets(test_graphs, test_labels)
            test_metrics = trainer.test(ext_test_loader)
        
        # Save results
        results[model_type] = {
            'trainer': trainer,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'config': config.copy()
        }
        
        # Plot training history
        trainer.plot_training_history(f'plots/{model_type}_training_history.png')
        
        print(f"{model_type.upper()} training completed in {training_time:.2f} seconds")
        
        # Save model
        trainer.save_model(f'{model_type}_final_model.pt')
    
    return results

def evaluate_on_attack_scenarios(models_results, attack_graphs, attack_labels, attack_info):
    """Evaluate models on specific attack scenarios.
    
    Args:
        models_results: Dictionary of trained models
        attack_graphs: Attack scenario graphs
        attack_labels: Attack scenario labels
        attack_info: Attack scenario information
        
    Returns:
        Evaluation results
    """
    print(f"\nEvaluating models on {len(attack_graphs)} attack scenarios...")
    
    attack_results = {}
    
    for model_type, model_data in models_results.items():
        trainer = model_data['trainer']
        model_results = defaultdict(list)
        
        print(f"\nEvaluating {model_type.upper()} model...")
        
        for i, (graph, labels, info) in enumerate(zip(attack_graphs, attack_labels, attack_info)):
            # Make predictions
            predictions = trainer.predict(graph)
            
            # Calculate attack detection metrics
            attackers = set(info['attackers'])
            
            # True labels: 1 for honest, 0 for attackers
            true_labels = []
            pred_scores = []
            
            for node_id in predictions.keys():
                if node_id in attackers:
                    true_labels.append(0)  # Attacker
                else:
                    true_labels.append(1)  # Honest
                pred_scores.append(predictions[node_id])
            
            # Binary predictions
            threshold = trainer.config['trust_threshold']
            pred_labels = [1 if score >= threshold else 0 for score in pred_scores]
            
            # Calculate metrics
            if len(set(true_labels)) > 1:  # Only if we have both classes
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(true_labels, pred_labels)
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                recall = recall_score(true_labels, pred_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_labels, zero_division=0)
                
                model_results['accuracy'].append(accuracy)
                model_results['precision'].append(precision)
                model_results['recall'].append(recall)
                model_results['f1'].append(f1)
                model_results['attack_type'].append(info['attack_type'])
        
        # Aggregate results
        attack_results[model_type] = {
            'mean_accuracy': np.mean(model_results['accuracy']) if model_results['accuracy'] else 0,
            'mean_precision': np.mean(model_results['precision']) if model_results['precision'] else 0,
            'mean_recall': np.mean(model_results['recall']) if model_results['recall'] else 0,
            'mean_f1': np.mean(model_results['f1']) if model_results['f1'] else 0,
            'detailed_results': dict(model_results)
        }
        
        print(f"{model_type.upper()} Attack Detection Results:")
        print(f"  Accuracy: {attack_results[model_type]['mean_accuracy']:.4f}")
        print(f"  Precision: {attack_results[model_type]['mean_precision']:.4f}")
        print(f"  Recall: {attack_results[model_type]['mean_recall']:.4f}")
        print(f"  F1-Score: {attack_results[model_type]['mean_f1']:.4f}")
    
    return attack_results

def plot_comparison_results(models_results, attack_results, save_dir='plots'):
    """Plot comparison of model performance.
    
    Args:
        models_results: Training results
        attack_results: Attack detection results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Training Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(models_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        values = [models_results[model]['test_metrics'].get(metric, 0) for model in models]
        
        bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax.set_title(f'Test {metric.capitalize()}')
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Attack Detection Performance
    if attack_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            values = [attack_results[model][f'mean_{metric}'] for model in models]
            
            bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            ax.set_title(f'Attack Detection {metric.capitalize()}')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/attack_detection_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot 3: Training Time Comparison
    plt.figure(figsize=(10, 6))
    training_times = [models_results[model]['training_time'] for model in models]
    bars = plt.bar(models, training_times, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Training Time Comparison')
    plt.ylabel('Training Time (seconds)')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(models_results, attack_results, save_path='results/gnn_trust_results.json'):
    """Save results to JSON file.
    
    Args:
        models_results: Training results
        attack_results: Attack detection results
        save_path: Path to save results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare data for JSON serialization
    json_results = {
        'training_results': {},
        'attack_detection_results': attack_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    for model_type, result in models_results.items():
        json_results['training_results'][model_type] = {
            'test_metrics': result['test_metrics'],
            'training_time': result['training_time'],
            'config': result['config']
        }
    
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {save_path}")

def main():
    """Main training and evaluation function."""
    print("🚀 Starting Comprehensive GNN Trust Training and Evaluation")
    print("="*70)
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints/gnn_trust', exist_ok=True)
    
    # Create environment
    print("📊 Setting up environment...")
    scenario = Scenario(config_file="examples/scenarios/configs/trust_config_1.json")
    env = Env_Trust(scenario, config_file="core/configs/env_config.json")
    
    # Generate training data
    print("\n📈 Generating training data...")
    graphs, trust_labels = create_synthetic_trust_data(env, num_samples=500)
    
    # Generate attack scenarios for evaluation
    print("\n🎯 Generating attack scenarios...")
    attack_graphs, attack_labels, attack_info = create_attack_scenarios(env, num_scenarios=100)
    
    # Train and evaluate models
    print("\n🧠 Training GNN models...")
    models_results = train_and_evaluate_models(graphs, trust_labels, attack_graphs, attack_labels)
    
    # Evaluate on attack scenarios
    print("\n🛡️ Evaluating attack detection performance...")
    attack_results = evaluate_on_attack_scenarios(models_results, attack_graphs, attack_labels, attack_info)
    
    # Generate comparison plots
    print("\n📊 Generating comparison plots...")
    plot_comparison_results(models_results, attack_results)
    
    # Save results
    print("\n💾 Saving results...")
    save_results(models_results, attack_results)
    
    # Print summary
    print("\n" + "="*70)
    print("🎉 Training and Evaluation Complete!")
    print("="*70)
    
    print("\n📈 FINAL RESULTS SUMMARY:")
    print("-" * 50)
    
    for model_type in models_results.keys():
        print(f"\n{model_type.upper()} Model:")
        test_metrics = models_results[model_type]['test_metrics']
        attack_perf = attack_results[model_type]
        
        print(f"  Trust Prediction:")
        print(f"    Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
        print(f"    Precision: {test_metrics.get('precision', 0):.4f}")
        print(f"    Recall:    {test_metrics.get('recall', 0):.4f}")
        print(f"    F1-Score:  {test_metrics.get('f1', 0):.4f}")
        
        print(f"  Attack Detection:")
        print(f"    Accuracy:  {attack_perf['mean_accuracy']:.4f}")
        print(f"    Precision: {attack_perf['mean_precision']:.4f}")
        print(f"    Recall:    {attack_perf['mean_recall']:.4f}")
        print(f"    F1-Score:  {attack_perf['mean_f1']:.4f}")
        
        print(f"  Training Time: {models_results[model_type]['training_time']:.1f}s")
    
    # Find best model
    best_model = max(models_results.keys(), 
                    key=lambda x: models_results[x]['test_metrics'].get('f1', 0))
    print(f"\n🏆 Best Model: {best_model.upper()}")
    print(f"   F1-Score: {models_results[best_model]['test_metrics'].get('f1', 0):.4f}")
    
    print("\n✅ All results saved to 'results/' and 'plots/' directories")
    print("✅ Model checkpoints saved to 'checkpoints/gnn_trust/' directory")

if __name__ == '__main__':
    main()