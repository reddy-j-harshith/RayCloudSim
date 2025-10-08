#!/usr/bin/env python3
"""
Improved Attack Simulation Demo using GNN Trust Regression

This demo showcases the fixed trust system:
1. Trust values are continuous [0,1] (regression, not classification)
2. Uses real topologies (pakistan/topo4mec) not synthetic data  
3. Malicious node detection via trust thresholds
4. Proper attack detection metrics
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import networkx as nx
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_gnn_trust_regression import TrustRegressionDataset, GNNTrustRegressor, TrustRegressionTrainer, detect_malicious_nodes_statistical

class ImprovedAttackSimulator:
    """Attack simulator using the new trust regression system"""
    
    def __init__(self, config_paths):
        self.config_paths = config_paths
        self.topologies = []
        self.models = {}
        self.attack_results = {}
        
        print("Initializing Improved Attack Simulator...")
        self._load_topologies()
        self._load_trained_models()
    
    def _load_topologies(self):
        """Load real network topologies"""
        print("Loading real network topologies...")
        
        for config_path in self.config_paths:
            if not os.path.exists(config_path):
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Build NetworkX graph
                G = nx.DiGraph()
                node_types = {}
                
                for node in config['Nodes']:
                    node_id = node['NodeId']
                    node_type = node['NodeType']
                    node_types[node_id] = node_type
                    
                    G.add_node(node_id, 
                              node_type=node_type,
                              cpu_freq=node.get('MaxCpuFreq', 1000),
                              buffer_size=node.get('MaxBufferSize', 100),
                              idle_energy=node.get('IdleEnergyCoef', 0.01),
                              exe_energy=node.get('ExeEnergyCoef', 0.1))
                
                for edge in config['Edges']:
                    src = edge['SrcNodeID']
                    dst = edge['DstNodeID']
                    bandwidth = edge.get('Bandwidth', 1000.0)
                    G.add_edge(src, dst, bandwidth=bandwidth)
                
                self.topologies.append({
                    'graph': G,
                    'node_types': node_types,
                    'config_name': os.path.basename(config_path),
                    'malicious_nodes': [nid for nid, ntype in node_types.items() if ntype == 'MaliciousNode']
                })
                
                print(f"Loaded {os.path.basename(config_path)}: {G.number_of_nodes()} nodes, "
                      f"{G.number_of_edges()} edges, {len([n for n in node_types.values() if n == 'MaliciousNode'])} malicious")
                
            except Exception as e:
                print(f"Error loading {config_path}: {e}")
    
    def _load_trained_models(self):
        """Load pre-trained GNN models"""
        print("Loading trained GNN models...")
        
        model_files = {
            'GAT': 'best_gat_trust_regressor.pth',
            'GraphSAGE': 'best_graphsage_trust_regressor.pth', 
            'GCN': 'best_gcn_trust_regressor.pth',
            'Transformer': 'best_transformer_trust_regressor.pth'
        }
        
        for model_name, filename in model_files.items():
            if os.path.exists(filename):
                try:
                    # Create model with same architecture
                    model = GNNTrustRegressor(input_dim=16, hidden_dim=128, model_type=model_name)
                    model.load_state_dict(torch.load(filename, map_location='cpu'))
                    model.eval()
                    
                    self.models[model_name] = model
                    print(f"Loaded {model_name} model")
                    
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}")
            else:
                print(f"Model file {filename} not found")
    
    def simulate_trust_based_attacks(self):
        """Simulate various attack scenarios using trust regression"""
        print("\nSimulating Trust-Based Attack Scenarios...")
        
        attack_scenarios = [
            {'name': 'Static Threshold', 'method': 'threshold', 'threshold': 0.5},
            {'name': 'Adaptive Threshold', 'method': 'threshold', 'threshold': 0.3},
            {'name': 'Statistical Z-Score', 'method': 'zscore', 'threshold': 2.0},
            {'name': 'IQR Outlier Detection', 'method': 'iqr', 'threshold': None},
            {'name': 'Bottom 20% Percentile', 'method': 'percentile', 'threshold': 20}
        ]
        
        results = {}
        
        for scenario in attack_scenarios:
            scenario_results = {}
            print(f"\nTesting {scenario['name']} detection...")
            
            for model_name, model in self.models.items():
                model_results = []
                
                for topo in self.topologies:
                    # Get trust predictions
                    trust_values = self._predict_trust_values(model, topo['graph'])
                    
                    # Detect malicious nodes using specified method
                    trust_vals = list(trust_values.values())
                    
                    if scenario['method'] == 'threshold':
                        # Simple threshold-based detection
                        threshold = scenario['threshold']
                        detected_malicious = [trust < threshold for trust in trust_vals]
                    else:
                        # Statistical methods
                        if scenario['threshold'] is not None:
                            detected_malicious = detect_malicious_nodes_statistical(
                                trust_vals, scenario['method'], scenario['threshold']
                            )
                        else:
                            detected_malicious = detect_malicious_nodes_statistical(
                                trust_vals, scenario['method']
                            )
                    
                    # Map back to node IDs
                    node_ids = list(trust_values.keys())
                    detected_nodes = [node_ids[i] for i, is_malicious in enumerate(detected_malicious) if is_malicious]
                    
                    # Calculate metrics
                    true_malicious = set(topo['malicious_nodes'])
                    detected_set = set(detected_nodes)
                    
                    tp = len(true_malicious & detected_set)
                    fp = len(detected_set - true_malicious)
                    fn = len(true_malicious - detected_set)
                    tn = len(node_ids) - tp - fp - fn
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    accuracy = (tp + tn) / len(node_ids)
                    
                    model_results.append({
                        'topology': topo['config_name'],
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'true_malicious': len(true_malicious),
                        'detected_malicious': len(detected_set),
                        'true_positives': tp,
                        'false_positives': fp,
                        'false_negatives': fn,
                        'avg_trust': np.mean(list(trust_values.values())),
                        'std_trust': np.std(list(trust_values.values()))
                    })
                
                scenario_results[model_name] = model_results
            
            results[scenario['name']] = scenario_results
        
        self.attack_results = results
        return results
    
    def _predict_trust_values(self, model, graph):
        """Predict trust values for a graph using trained model"""
        try:
            node_ids = list(graph.nodes())
            if not node_ids:
                return {}
            
            # Manually convert graph to model input (same as TrustRegressionDataset)
            node_features = []
            node_types = {}
            
            # Extract node types from graph
            for node_id in node_ids:
                node_data = graph.nodes[node_id]
                node_types[node_id] = node_data.get('node_type', 'TrustNode')
            
            # Compute features for each node
            for node_id in node_ids:
                features = self._compute_node_features(graph, node_id, node_types)
                node_features.append(features)
            
            # Create edge indices
            edge_list = list(graph.edges())
            if not edge_list:
                edge_list = [(node, node) for node in node_ids]
            
            edge_index = []
            for src, dst in edge_list:
                src_idx = node_ids.index(src)
                dst_idx = node_ids.index(dst)
                edge_index.append([src_idx, dst_idx])
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Predict
            with torch.no_grad():
                trust_outputs = model(x, edge_index)
                
            # Map back to node IDs
            trust_dict = {}
            if trust_outputs.dim() == 0:  # Single node
                trust_dict[node_ids[0]] = trust_outputs.item()
            else:
                for i, node_id in enumerate(node_ids):
                    trust_dict[node_id] = trust_outputs[i].item()
            
            return trust_dict
            
        except Exception as e:
            print(f"Error predicting trust values: {e}")
            return {node_id: 0.5 for node_id in graph.nodes()}
    
    def _compute_node_features(self, G, node_id, node_types):
        """Compute comprehensive node features (same as in training)"""
        node_data = G.nodes[node_id]
        
        # Basic node properties
        cpu_freq = node_data.get('cpu_freq', 1000) / 100000.0  # Normalize
        buffer_size = node_data.get('buffer_size', 100) / 1000.0  # Normalize
        idle_energy = node_data.get('idle_energy', 0.01)
        exe_energy = node_data.get('exe_energy', 0.1)
        
        # Network centrality measures
        try:
            degree_cent = nx.degree_centrality(G)[node_id]
            betweenness_cent = nx.betweenness_centrality(G).get(node_id, 0.0)
            closeness_cent = nx.closeness_centrality(G).get(node_id, 0.0)
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000).get(node_id, 0.0)
        except:
            degree_cent = betweenness_cent = closeness_cent = eigenvector_cent = 0.0
        
        # Local network properties
        degree = G.degree(node_id)
        in_degree = G.in_degree(node_id) if G.is_directed() else degree
        out_degree = G.out_degree(node_id) if G.is_directed() else degree
        
        try:
            clustering = nx.clustering(G.to_undirected())[node_id]
        except:
            clustering = 0.0
        
        # Node type encoding
        node_type = node_types.get(node_id, 'TrustNode')
        type_encoding = {
            'TrustNode': [1, 0, 0, 0],
            'GNNTrustNode': [0, 1, 0, 0], 
            'MaliciousNode': [0, 0, 1, 0],
            'Edge': [0, 0, 0, 1],
            'Fog': [0, 0, 0, 1],
            'Cloud': [0, 0, 0, 1]
        }.get(node_type, [0, 0, 0, 1])
        
        # Combine all features
        features = [
            cpu_freq, buffer_size, idle_energy, exe_energy,
            degree_cent, betweenness_cent, closeness_cent, eigenvector_cent,
            degree / 10.0, in_degree / 10.0, out_degree / 10.0, clustering
        ] + type_encoding
        
        return np.array(features, dtype=np.float32)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive attack simulation report"""
        print("\nGenerating Comprehensive Attack Simulation Report...")
        
        # Create results directory
        results_dir = f"attack_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Overall Performance Summary
        summary_data = []
        
        for scenario_name, scenario_results in self.attack_results.items():
            for model_name, model_results in scenario_results.items():
                avg_metrics = {
                    'scenario': scenario_name,
                    'model': model_name,
                    'avg_accuracy': np.mean([r['accuracy'] for r in model_results]),
                    'avg_precision': np.mean([r['precision'] for r in model_results]),
                    'avg_recall': np.mean([r['recall'] for r in model_results]),
                    'avg_f1': np.mean([r['f1'] for r in model_results]),
                    'avg_trust': np.mean([r['avg_trust'] for r in model_results]),
                    'std_trust': np.mean([r['std_trust'] for r in model_results])
                }
                summary_data.append(avg_metrics)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(results_dir, 'attack_simulation_summary.csv'), index=False)
        
        # 2. Detailed Results by Topology
        detailed_data = []
        
        for scenario_name, scenario_results in self.attack_results.items():
            for model_name, model_results in scenario_results.items():
                for result in model_results:
                    result['scenario'] = scenario_name
                    result['model'] = model_name
                    detailed_data.append(result)
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(os.path.join(results_dir, 'detailed_attack_results.csv'), index=False)
        
        # 3. Visualization
        self._create_attack_visualizations(summary_df, detailed_df, results_dir)
        
        # 4. Analysis Report
        self._generate_analysis_report(summary_df, detailed_df, results_dir)
        
        print(f"Report generated in: {results_dir}")
        return results_dir
    
    def _create_attack_visualizations(self, summary_df, detailed_df, results_dir):
        """Create comprehensive visualizations"""
        
        # 1. Performance Comparison Heatmap
        plt.figure(figsize=(15, 10))
        
        # Pivot for heatmap
        pivot_data = summary_df.pivot_table(
            values='avg_f1', 
            index='scenario', 
            columns='model',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Average F1 Score'})
        plt.title('Attack Detection Performance: F1 Score by Scenario and Model')
        plt.xlabel('GNN Model')
        plt.ylabel('Detection Scenario')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Comparison Radar Chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        metrics = ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1']
        models = summary_df['model'].unique()
        
        for i, model in enumerate(models):
            ax = axes[i]
            model_data = summary_df[summary_df['model'] == model]
            
            # Average across scenarios
            avg_metrics = [model_data[metric].mean() for metric in metrics]
            
            # Close the plot
            avg_metrics.append(avg_metrics[0])
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles.append(angles[0])
            
            ax.plot(angles, avg_metrics, 'o-', linewidth=2, label=model)
            ax.fill(angles, avg_metrics, alpha=0.25)
            ax.set_ylim(0, 1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
            ax.set_title(f'{model} Model Performance')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'model_radar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scenario Performance Analysis
        plt.figure(figsize=(15, 8))
        
        scenario_perf = summary_df.groupby('scenario').agg({
            'avg_accuracy': 'mean',
            'avg_precision': 'mean', 
            'avg_recall': 'mean',
            'avg_f1': 'mean'
        }).reset_index()
        
        x = range(len(scenario_perf))
        width = 0.2
        
        plt.bar([i - 1.5*width for i in x], scenario_perf['avg_accuracy'], width, label='Accuracy', alpha=0.8)
        plt.bar([i - 0.5*width for i in x], scenario_perf['avg_precision'], width, label='Precision', alpha=0.8)
        plt.bar([i + 0.5*width for i in x], scenario_perf['avg_recall'], width, label='Recall', alpha=0.8)
        plt.bar([i + 1.5*width for i in x], scenario_perf['avg_f1'], width, label='F1 Score', alpha=0.8)
        
        plt.xlabel('Detection Scenario')
        plt.ylabel('Performance Score')
        plt.title('Attack Detection Performance by Scenario')
        plt.xticks(x, scenario_perf['scenario'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'scenario_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Trust Distribution Analysis
        plt.figure(figsize=(15, 10))
        
        trust_data = []
        for scenario_name, scenario_results in self.attack_results.items():
            for model_name, model_results in scenario_results.items():
                for result in model_results:
                    trust_data.append({
                        'scenario': scenario_name,
                        'model': model_name,
                        'topology': result['topology'],
                        'avg_trust': result['avg_trust'],
                        'std_trust': result['std_trust']
                    })
        
        trust_df = pd.DataFrame(trust_data)
        
        plt.subplot(2, 2, 1)
        sns.boxplot(data=trust_df, x='model', y='avg_trust')
        plt.title('Average Trust Distribution by Model')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        sns.boxplot(data=trust_df, x='scenario', y='avg_trust')
        plt.title('Average Trust Distribution by Scenario')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=trust_df, x='avg_trust', y='std_trust', hue='model')
        plt.title('Trust Mean vs Standard Deviation')
        
        plt.subplot(2, 2, 4)
        topology_trust = trust_df.groupby('topology')['avg_trust'].mean().sort_values()
        plt.bar(range(len(topology_trust)), topology_trust.values)
        plt.xticks(range(len(topology_trust)), [t.split('_')[1] if '_' in t else t for t in topology_trust.index], rotation=45)
        plt.title('Average Trust by Topology')
        plt.ylabel('Average Trust Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'trust_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_analysis_report(self, summary_df, detailed_df, results_dir):
        """Generate detailed analysis report"""
        
        report_path = os.path.join(results_dir, 'attack_simulation_analysis.md')
        
        with open(report_path, 'w') as f:
            f.write("# GNN Trust Regression Attack Simulation Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes the performance of GNN-based trust regression models ")
            f.write("for malicious node detection across different attack scenarios and network topologies.\n\n")
            
            # Best performing combinations
            best_combo = summary_df.loc[summary_df['avg_f1'].idxmax()]
            f.write(f"**Best Performance**: {best_combo['model']} with {best_combo['scenario']} ")
            f.write(f"(F1: {best_combo['avg_f1']:.4f})\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Model performance
            model_perf = summary_df.groupby('model')['avg_f1'].mean().sort_values(ascending=False)
            f.write("### Model Performance Ranking\n")
            for i, (model, f1) in enumerate(model_perf.items(), 1):
                f.write(f"{i}. **{model}**: {f1:.4f} average F1 score\n")
            
            f.write("\n### Scenario Effectiveness\n")
            scenario_perf = summary_df.groupby('scenario')['avg_f1'].mean().sort_values(ascending=False)
            for i, (scenario, f1) in enumerate(scenario_perf.items(), 1):
                f.write(f"{i}. **{scenario}**: {f1:.4f} average F1 score\n")
            
            f.write("\n## Detailed Analysis\n\n")
            
            # Trust value analysis
            avg_trust = detailed_df['avg_trust'].mean()
            std_trust = detailed_df['std_trust'].mean()
            f.write(f"- **Average Trust Value**: {avg_trust:.4f}\n")
            f.write(f"- **Average Trust Std Dev**: {std_trust:.4f}\n")
            
            # Detection statistics
            total_malicious = detailed_df['true_malicious'].sum()
            total_detected = detailed_df['detected_malicious'].sum()
            total_tp = detailed_df['true_positives'].sum()
            total_fp = detailed_df['false_positives'].sum()
            
            f.write(f"- **Total Malicious Nodes**: {total_malicious}\n")
            f.write(f"- **Total Detections**: {total_detected}\n")
            f.write(f"- **True Positives**: {total_tp}\n")
            f.write(f"- **False Positives**: {total_fp}\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("1. **Model Selection**: Use the best performing model for production deployment\n")
            f.write("2. **Threshold Tuning**: Fine-tune detection thresholds based on network characteristics\n")
            f.write("3. **Ensemble Methods**: Consider combining multiple detection scenarios\n")
            f.write("4. **Continuous Learning**: Retrain models periodically with new attack patterns\n")
            
            f.write("\n## Technical Notes\n\n")
            f.write("- Trust values are continuous in [0,1] range (regression approach)\n")
            f.write("- Real network topologies used (pakistan, topo4mec)\n")
            f.write("- Multiple detection methods evaluated (threshold, statistical)\n")
            f.write("- Performance metrics: Accuracy, Precision, Recall, F1-Score\n")

def main():
    """Main function to run improved attack simulation"""
    print("Improved GNN Trust Attack Simulation Demo")
    print("=" * 50)
    
    # Configuration paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(base_dir, "experiments/gnn_trust/gnn_pakistan_Tuple30K_config.json"),
        os.path.join(base_dir, "experiments/gnn_trust/gnn_pakistan_Tuple50K_config.json"), 
        os.path.join(base_dir, "experiments/gnn_trust/gnn_pakistan_Tuple100K_config.json"),
        os.path.join(base_dir, "experiments/gnn_trust/gnn_topo4mec_25N50E_config.json"),
        os.path.join(base_dir, "experiments/gnn_trust/gnn_topo4mec_50N50E_config.json"),
        os.path.join(base_dir, "experiments/gnn_trust/gnn_topo4mec_100N150E_config.json")
    ]
    
    # Filter existing configs
    existing_configs = [path for path in config_paths if os.path.exists(path)]
    print(f"Found {len(existing_configs)} topology configurations")
    
    if not existing_configs:
        print("No topology configurations found!")
        return
    
    # Initialize simulator
    simulator = ImprovedAttackSimulator(existing_configs)
    
    if not simulator.models:
        print("No trained models found! Please run train_gnn_trust_regression.py first.")
        return
    
    # Run attack simulation
    results = simulator.simulate_trust_based_attacks()
    
    # Generate comprehensive report
    report_dir = simulator.generate_comprehensive_report()
    
    print(f"\nAttack simulation completed!")
    print(f"Results saved in: {report_dir}")
    
    # Print summary
    print("\nQuick Summary:")
    for scenario_name, scenario_results in results.items():
        print(f"\n{scenario_name}:")
        for model_name, model_results in scenario_results.items():
            avg_f1 = np.mean([r['f1'] for r in model_results])
            avg_acc = np.mean([r['accuracy'] for r in model_results])
            print(f"  {model_name}: F1={avg_f1:.4f}, Acc={avg_acc:.4f}")

if __name__ == "__main__":
    main()