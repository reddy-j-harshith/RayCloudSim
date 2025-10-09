#!/usr/bin/env python3
"""
Enhanced Visualization System for GNN Trust Analysis
Creates detailed plots for trust trajectories, offloading performance, and attack analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from datetime import datetime

class EnhancedVisualizationSystem:
    """Enhanced visualization system for comprehensive trust analysis"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'enhanced_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
    def create_individual_gnn_trust_trajectories(self, results_data: Dict, dataset_key: str, malicious_nodes: List[int]):
        """Create separate trust trajectory plots for each GNN model"""
        print(f"   📊 Creating individual GNN trust trajectories for {dataset_key}")
        
        # Extract GNN results
        gnn_models = ['GAT', 'GraphSAGE', 'GCN', 'Transformer']
        
        # Get temporal trust data
        train_results = results_data.get('training_results', {})
        temporal_data = train_results.get('temporal_trust_data', [])
        
        if not temporal_data:
            print(f"   ⚠️ No temporal trust data found for {dataset_key}")
            return
        
        df = pd.DataFrame(temporal_data)
        
        # Create individual plots for each GNN model
        for model_name in gnn_models:
            if model_name in results_data.get('gnn_results', {}):
                self._plot_single_gnn_trajectory(df, model_name, dataset_key, malicious_nodes)
    
    def _plot_single_gnn_trajectory(self, df: pd.DataFrame, model_name: str, dataset_key: str, malicious_nodes: List[int]):
        """Plot trust trajectory for a single GNN model"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_name} Trust Analysis - {dataset_key}', fontsize=16, fontweight='bold')
        
        # Plot 1: Trust evolution over time (malicious vs honest)
        ax1 = axes[0, 0]
        
        # Separate malicious and honest nodes
        malicious_data = df[df['node_id'].isin(malicious_nodes)]
        honest_data = df[~df['node_id'].isin(malicious_nodes)]
        
        if not malicious_data.empty:
            malicious_avg = malicious_data.groupby('task_index')['avg_trust'].mean()
            ax1.plot(malicious_avg.index, malicious_avg.values, 'r-', linewidth=2, 
                    label=f'Malicious Nodes (n={len(malicious_nodes)})', alpha=0.8)
        
        if not honest_data.empty:
            honest_avg = honest_data.groupby('task_index')['avg_trust'].mean()
            ax1.plot(honest_avg.index, honest_avg.values, 'b-', linewidth=2, 
                    label=f'Honest Nodes (n={len(df["node_id"].unique()) - len(malicious_nodes)})', alpha=0.8)
        
        ax1.set_xlabel('Task Index')
        ax1.set_ylabel('Average Trust Score')
        ax1.set_title(f'{model_name}: Trust Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual node trajectories
        ax2 = axes[0, 1]
        
        for node_id in df['node_id'].unique()[:10]:  # Limit to first 10 nodes for clarity
            node_data = df[df['node_id'] == node_id]
            color = 'red' if node_id in malicious_nodes else 'blue'
            alpha = 0.7 if node_id in malicious_nodes else 0.5
            ax2.plot(node_data['task_index'], node_data['avg_trust'], 
                    color=color, alpha=alpha, linewidth=1)
        
        ax2.set_xlabel('Task Index')
        ax2.set_ylabel('Trust Score')
        ax2.set_title(f'{model_name}: Individual Node Trajectories')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final trust distribution
        ax3 = axes[1, 0]
        
        # Get final trust values
        final_trust = df.groupby('node_id')['avg_trust'].last()
        
        malicious_final = final_trust[final_trust.index.isin(malicious_nodes)]
        honest_final = final_trust[~final_trust.index.isin(malicious_nodes)]
        
        ax3.hist(malicious_final.values, bins=15, alpha=0.7, color='red', 
                label=f'Malicious (median={np.median(malicious_final.values):.3f})', density=True)
        ax3.hist(honest_final.values, bins=15, alpha=0.7, color='blue', 
                label=f'Honest (median={np.median(honest_final.values):.3f})', density=True)
        
        ax3.axvline(np.median(malicious_final.values), color='darkred', linestyle='--', alpha=0.8)
        ax3.axvline(np.median(honest_final.values), color='darkblue', linestyle='--', alpha=0.8)
        
        ax3.set_xlabel('Final Trust Score')
        ax3.set_ylabel('Density')
        ax3.set_title(f'{model_name}: Final Trust Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trust statistics over time
        ax4 = axes[1, 1]
        
        # Calculate rolling statistics
        window_size = max(10, len(df) // 20)
        
        trust_stats = df.groupby('task_index')['avg_trust'].agg(['mean', 'std', 'median']).rolling(window=window_size).mean()
        
        ax4.plot(trust_stats.index, trust_stats['mean'], 'g-', linewidth=2, label='Mean Trust')
        ax4.fill_between(trust_stats.index, 
                        trust_stats['mean'] - trust_stats['std'], 
                        trust_stats['mean'] + trust_stats['std'], 
                        alpha=0.3, color='green')
        ax4.plot(trust_stats.index, trust_stats['median'], 'orange', linewidth=2, label='Median Trust')
        
        ax4.set_xlabel('Task Index')
        ax4.set_ylabel('Trust Score')
        ax4.set_title(f'{model_name}: Trust Statistics Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name}_trust_trajectory_{dataset_key}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_trust_based_offloading_analysis(self, results_data: Dict, dataset_key: str, malicious_nodes: List[int]):
        """Create comprehensive trust-based offloading analysis"""
        print(f"   📊 Creating trust-based offloading analysis for {dataset_key}")
        
        train_results = results_data.get('training_results', {})
        test_results = results_data.get('testing_results', {})
        
        # Get task logs
        train_logs = train_results.get('phase_logger', {}).get('task_logs', [])
        test_logs = test_results.get('phase_logger', {}).get('task_logs', [])
        
        if not train_logs and not test_logs:
            print(f"   ⚠️ No task logs found for {dataset_key}")
            return
        
        # Combine logs
        all_logs = train_logs + test_logs
        df = pd.DataFrame(all_logs)
        
        if df.empty:
            return
        
        # Create comprehensive offloading analysis
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Trust-Based Offloading Analysis - {dataset_key}', fontsize=16, fontweight='bold')
        
        # Plot 1: Success rate over time with/without trust-based offloading
        ax1 = axes[0, 0]
        self._plot_offloading_effectiveness(df, ax1, malicious_nodes)
        
        # Plot 2: Trust score vs success rate
        ax2 = axes[0, 1]
        self._plot_trust_vs_success(df, ax2)
        
        # Plot 3: Attack impact analysis
        ax3 = axes[1, 0]
        self._plot_attack_impact(df, ax3, malicious_nodes)
        
        # Plot 4: Trust distribution during attacks
        ax4 = axes[1, 1]
        self._plot_trust_during_attacks(df, ax4, malicious_nodes)
        
        # Plot 5: Offloading pattern analysis
        ax5 = axes[2, 0]
        self._plot_offloading_patterns(df, ax5, malicious_nodes)
        
        # Plot 6: Protection effectiveness
        ax6 = axes[2, 1]
        self._plot_protection_effectiveness(df, ax6, malicious_nodes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'offloading_analysis_{dataset_key}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_offloading_effectiveness(self, df: pd.DataFrame, ax, malicious_nodes: List[int]):
        """Plot effectiveness of trust-based offloading"""
        # Simulate what would happen without trust-based offloading (random selection)
        np.random.seed(42)
        df_copy = df.copy()
        
        # Calculate success rates with trust-based offloading (actual)
        window_size = max(10, len(df) // 20)
        df_copy['timestamp_bin'] = pd.cut(df_copy['timestamp'], bins=50)
        
        actual_success = df_copy.groupby('timestamp_bin')['execution_success'].mean()
        
        # Simulate random offloading (without trust)
        total_nodes = len(set(df['dst_node'].unique()))
        malicious_ratio = len(malicious_nodes) / total_nodes
        
        # Random offloading would have success rate = (1 - malicious_ratio) assuming malicious always fail
        random_success_rate = 1 - malicious_ratio
        
        x_range = range(len(actual_success))
        ax.plot(x_range, actual_success.values, 'b-', linewidth=3, 
               label=f'With Trust-Based Offloading (Avg: {actual_success.mean():.3f})')
        ax.axhline(y=random_success_rate, color='red', linestyle='--', linewidth=2, 
                  label=f'Without Trust (Random: {random_success_rate:.3f})')
        
        ax.fill_between(x_range, actual_success.values, random_success_rate, 
                       where=(actual_success.values > random_success_rate), 
                       color='green', alpha=0.3, label='Improvement')
        
        ax.set_xlabel('Time Periods')
        ax.set_ylabel('Success Rate')
        ax.set_title('Trust-Based Offloading Effectiveness')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trust_vs_success(self, df: pd.DataFrame, ax):
        """Plot trust score vs success rate correlation"""
        # Bin trust scores and calculate success rates
        df['trust_bin'] = pd.cut(df['trust_score'], bins=20)
        trust_success = df.groupby('trust_bin')['execution_success'].agg(['mean', 'count'])
        
        # Get bin centers for plotting
        bin_centers = [interval.mid for interval in trust_success.index]
        
        # Plot with size proportional to sample count
        scatter = ax.scatter(bin_centers, trust_success['mean'], 
                           s=trust_success['count']*2, alpha=0.6, c=trust_success['mean'], 
                           cmap='RdYlBu', edgecolors='black')
        
        # Add trend line
        z = np.polyfit(bin_centers, trust_success['mean'], 1)
        p = np.poly1d(z)
        ax.plot(bin_centers, p(bin_centers), "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Trust Score')
        ax.set_ylabel('Success Rate')
        ax.set_title('Trust Score vs Task Success Rate')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Success Rate')
    
    def _plot_attack_impact(self, df: pd.DataFrame, ax, malicious_nodes: List[int]):
        """Plot impact of attacks on system performance"""
        # Calculate attack intensity over time
        df['timestamp_bin'] = pd.cut(df['timestamp'], bins=30)
        
        attack_stats = df.groupby('timestamp_bin').agg({
            'is_dst_malicious': 'sum',  # Number of tasks sent to malicious nodes
            'execution_success': 'mean',  # Overall success rate
            'dst_node': 'count'  # Total tasks
        })
        
        attack_stats['attack_intensity'] = attack_stats['is_dst_malicious'] / attack_stats['dst_node']
        
        # Plot attack intensity and success rate
        ax2 = ax.twinx()
        
        x_range = range(len(attack_stats))
        line1 = ax.plot(x_range, attack_stats['attack_intensity'], 'r-', linewidth=2, 
                       label='Attack Intensity', alpha=0.8)
        line2 = ax2.plot(x_range, attack_stats['execution_success'], 'b-', linewidth=2, 
                        label='Success Rate', alpha=0.8)
        
        ax.set_xlabel('Time Periods')
        ax.set_ylabel('Attack Intensity (Fraction to Malicious)', color='red')
        ax2.set_ylabel('Success Rate', color='blue')
        ax.set_title('Attack Impact on System Performance')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_trust_during_attacks(self, df: pd.DataFrame, ax, malicious_nodes: List[int]):
        """Plot trust distribution during high/low attack periods"""
        # Identify attack periods
        df['timestamp_bin'] = pd.cut(df['timestamp'], bins=20)
        attack_intensity = df.groupby('timestamp_bin')['is_dst_malicious'].mean()
        
        high_attack_threshold = attack_intensity.quantile(0.75)
        low_attack_threshold = attack_intensity.quantile(0.25)
        
        # Get trust scores during different attack periods
        high_attack_bins = attack_intensity[attack_intensity >= high_attack_threshold].index
        low_attack_bins = attack_intensity[attack_intensity <= low_attack_threshold].index
        
        high_attack_trust = df[df['timestamp_bin'].isin(high_attack_bins)]['trust_score']
        low_attack_trust = df[df['timestamp_bin'].isin(low_attack_bins)]['trust_score']
        
        # Plot distributions
        ax.hist(low_attack_trust, bins=25, alpha=0.7, color='blue', density=True,
               label=f'Low Attack Periods (median={np.median(low_attack_trust):.3f})')
        ax.hist(high_attack_trust, bins=25, alpha=0.7, color='red', density=True,
               label=f'High Attack Periods (median={np.median(high_attack_trust):.3f})')
        
        # Add median lines
        ax.axvline(np.median(low_attack_trust), color='darkblue', linestyle='--', alpha=0.8)
        ax.axvline(np.median(high_attack_trust), color='darkred', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Trust Score')
        ax.set_ylabel('Density')
        ax.set_title('Trust Distribution: High vs Low Attack Periods')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_offloading_patterns(self, df: pd.DataFrame, ax, malicious_nodes: List[int]):
        """Plot offloading patterns to different node types"""
        # Calculate offloading statistics
        node_stats = df.groupby('dst_node').agg({
            'execution_success': ['mean', 'count'],
            'trust_score': 'mean',
            'is_dst_malicious': 'first'
        }).round(3)
        
        node_stats.columns = ['success_rate', 'task_count', 'avg_trust', 'is_malicious']
        
        # Separate malicious and honest nodes
        malicious_stats = node_stats[node_stats['is_malicious'] == True]
        honest_stats = node_stats[node_stats['is_malicious'] == False]
        
        # Plot task distribution
        ax.scatter(honest_stats['avg_trust'], honest_stats['success_rate'], 
                  s=honest_stats['task_count']*2, alpha=0.6, color='blue', 
                  label=f'Honest Nodes (n={len(honest_stats)})')
        ax.scatter(malicious_stats['avg_trust'], malicious_stats['success_rate'], 
                  s=malicious_stats['task_count']*2, alpha=0.6, color='red', 
                  label=f'Malicious Nodes (n={len(malicious_stats)})')
        
        ax.set_xlabel('Average Trust Score')
        ax.set_ylabel('Success Rate')
        ax.set_title('Node Performance vs Trust (Bubble size = Task Count)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_protection_effectiveness(self, df: pd.DataFrame, ax, malicious_nodes: List[int]):
        """Plot protection effectiveness metrics"""
        # Calculate protection metrics
        total_tasks = len(df)
        tasks_to_malicious = len(df[df['is_dst_malicious'] == True])
        successful_tasks = len(df[df['execution_success'] == True])
        
        # Calculate what would happen with random offloading
        malicious_ratio = len(malicious_nodes) / len(set(df['dst_node'].unique()))
        expected_malicious_tasks = total_tasks * malicious_ratio
        
        # Protection metrics
        protection_rate = 1 - (tasks_to_malicious / expected_malicious_tasks) if expected_malicious_tasks > 0 else 1
        efficiency = successful_tasks / total_tasks
        
        # Create bar chart
        metrics = ['Protection Rate', 'Task Success Rate', 'Malicious Avoidance']
        values = [
            protection_rate,
            efficiency,
            1 - (tasks_to_malicious / total_tasks)
        ]
        
        colors = ['green', 'blue', 'orange']
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Rate')
        ax.set_title('Trust-Based Protection Effectiveness')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_comprehensive_summary_plots(self, all_results: Dict):
        """Create summary plots across all datasets and models"""
        print("   📊 Creating comprehensive summary plots")
        
        # Collect data across all datasets and models
        model_performance = {}
        dataset_performance = {}
        
        for dataset_key, dataset_results in all_results.items():
            dataset_performance[dataset_key] = {}
            
            for model_type in ['GAT', 'GraphSAGE', 'GCN', 'Transformer']:
                if model_type not in model_performance:
                    model_performance[model_type] = {}
                
                # Get performance metrics
                metrics = dataset_results.get('final_metrics', {})
                downstream_f1 = metrics.get('downstream_f1', 0)
                detection_f1 = metrics.get('detection_f1', 0)
                offloading_efficiency = metrics.get('offloading_efficiency', 0)
                
                model_performance[model_type][dataset_key] = {
                    'downstream_f1': downstream_f1,
                    'detection_f1': detection_f1,
                    'offloading_efficiency': offloading_efficiency
                }
                
                dataset_performance[dataset_key][model_type] = {
                    'downstream_f1': downstream_f1,
                    'detection_f1': detection_f1,
                    'offloading_efficiency': offloading_efficiency
                }
        
        # Create summary plots
        self._plot_model_comparison_heatmap(model_performance)
        self._plot_dataset_performance_radar(dataset_performance)
        self._plot_performance_trends(model_performance)
    
    def _plot_model_comparison_heatmap(self, model_performance: Dict):
        """Create heatmap comparing model performance across datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['downstream_f1', 'detection_f1', 'offloading_efficiency']
        metric_names = ['Downstream F1', 'Detection F1', 'Offloading Efficiency']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            # Create data matrix
            models = list(model_performance.keys())
            datasets = list(model_performance[models[0]].keys()) if models else []
            
            data = np.zeros((len(models), len(datasets)))
            
            for i, model in enumerate(models):
                for j, dataset in enumerate(datasets):
                    data[i, j] = model_performance[model][dataset][metric]
            
            # Create heatmap
            sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       xticklabels=datasets, yticklabels=models, 
                       ax=axes[idx], cbar_kws={'label': name})
            axes[idx].set_title(f'{name} Comparison')
            axes[idx].set_xlabel('Datasets')
            axes[idx].set_ylabel('Models')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_performance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dataset_performance_radar(self, dataset_performance: Dict):
        """Create radar plots for dataset performance"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        metrics = ['downstream_f1', 'detection_f1', 'offloading_efficiency']
        
        for idx, (dataset, perf_data) in enumerate(dataset_performance.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Prepare data for radar chart
            models = list(perf_data.keys())
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for model in models:
                values = [perf_data[model][metric] for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(['Downstream F1', 'Detection F1', 'Offloading Eff.'])
            ax.set_ylim(0, 1)
            ax.set_title(f'{dataset} Performance', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)
        
        # Hide unused subplots
        for idx in range(len(dataset_performance), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'dataset_performance_radar.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_trends(self, model_performance: Dict):
        """Plot performance trends across different metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect all data
        all_data = []
        for model, datasets in model_performance.items():
            for dataset, metrics in datasets.items():
                all_data.append({
                    'model': model,
                    'dataset': dataset,
                    'downstream_f1': metrics['downstream_f1'],
                    'detection_f1': metrics['detection_f1'],
                    'offloading_efficiency': metrics['offloading_efficiency']
                })
        
        df = pd.DataFrame(all_data)
        
        # Plot 1: Model performance comparison
        ax1 = axes[0, 0]
        df.groupby('model')[['downstream_f1', 'detection_f1', 'offloading_efficiency']].mean().plot(
            kind='bar', ax=ax1, color=['blue', 'red', 'green'], alpha=0.7)
        ax1.set_title('Average Performance by Model')
        ax1.set_ylabel('Score')
        ax1.legend(['Downstream F1', 'Detection F1', 'Offloading Eff.'])
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Dataset difficulty analysis
        ax2 = axes[0, 1]
        df.groupby('dataset')[['downstream_f1', 'detection_f1', 'offloading_efficiency']].mean().plot(
            kind='bar', ax=ax2, color=['blue', 'red', 'green'], alpha=0.7)
        ax2.set_title('Average Performance by Dataset')
        ax2.set_ylabel('Score')
        ax2.legend(['Downstream F1', 'Detection F1', 'Offloading Eff.'])
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Correlation analysis
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['downstream_f1'], df['detection_f1'], 
                            c=df['offloading_efficiency'], s=100, alpha=0.7, cmap='viridis')
        ax3.set_xlabel('Downstream F1')
        ax3.set_ylabel('Detection F1')
        ax3.set_title('Performance Correlation Analysis')
        plt.colorbar(scatter, ax=ax3, label='Offloading Efficiency')
        
        # Plot 4: Performance variance
        ax4 = axes[1, 1]
        performance_std = df.groupby('model')[['downstream_f1', 'detection_f1', 'offloading_efficiency']].std()
        performance_std.plot(kind='bar', ax=ax4, color=['blue', 'red', 'green'], alpha=0.7)
        ax4.set_title('Performance Variance by Model')
        ax4.set_ylabel('Standard Deviation')
        ax4.legend(['Downstream F1', 'Detection F1', 'Offloading Eff.'])
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'performance_trends.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_enhanced_plots(self, all_results: Dict):
        """Generate all enhanced plots for the evaluation results"""
        print("🎨 Generating enhanced visualization plots...")
        
        for dataset_key, results_data in all_results.items():
            print(f"   📊 Processing {dataset_key}")
            
            # Extract malicious nodes info
            train_results = results_data.get('training_results', {})
            malicious_nodes = train_results.get('malicious_nodes', [])
            
            # Create individual GNN trust trajectories
            self.create_individual_gnn_trust_trajectories(results_data, dataset_key, malicious_nodes)
            
            # Create trust-based offloading analysis
            self.create_trust_based_offloading_analysis(results_data, dataset_key, malicious_nodes)
        
        # Create comprehensive summary plots
        self.create_comprehensive_summary_plots(all_results)
        
        print(f"✅ Enhanced plots saved to: {self.plots_dir}")
        return self.plots_dir