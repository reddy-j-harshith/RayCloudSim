#!/usr/bin/env python3
"""
🔧 Edge Computing-Focused GNN Trust Retraining System

This module retrains and evaluates GNN models with a focus on edge computing
scenarios based on the Pakistan dataset which has rich edge-fog-cloud hierarchies.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GATConv, SAGEConv, GCNConv, TransformerConv, GINConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch Geometric not available. Creating simplified models.")
    TORCH_GEOMETRIC_AVAILABLE = False

class EdgeGNNRetrainingSystem:
    """Advanced GNN Trust System for Edge Computing with Dynamic Retraining"""
    
    def __init__(self, dataset_name="PAKISTAN_Tuple30K"):
        self.dataset_name = dataset_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"edge_retraining_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🔧 Edge GNN Retraining System Initialized")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        # Initialize components
        self.simulator = None
        self.dataset_loader = None
        self.trust_system = None
        self.models = {}
        self.results = {}
        
        # Model configurations optimized for edge scenarios
        self.model_configs = {
            'GCN': {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.3},
            'GAT': {'hidden_dim': 64, 'num_heads': 4, 'num_layers': 3, 'dropout': 0.3},
            'GraphSAGE': {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.3},
            'GIN': {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.3}
        }
        
    def initialize_system(self):
        """Initialize the simulation system with Pakistan dataset"""
        try:
            # Load dataset
            self.dataset_loader = DatasetLoader()
            train_tasks, test_tasks, dataset_info = self.dataset_loader.load_dataset(self.dataset_name)
            
            print(f"📊 Dataset Loaded: {self.dataset_name}")
            print(f"   📋 Training tasks: {len(train_tasks):,}")
            print(f"   📋 Testing tasks: {len(test_tasks):,}")
            print(f"   🌐 Network nodes: {len(dataset_info['Nodes'])}")
            
            # Initialize simulator with dataset info
            self.simulator = RayCloudSim(
                nodes=dataset_info['Nodes'],
                edges=dataset_info['Edges']
            )
            
            # Initialize trust system
            self.trust_system = TrustSystem(self.simulator)
            
            # Store data
            self.train_tasks = train_tasks
            self.test_tasks = test_tasks
            self.dataset_info = dataset_info
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            return False
    
    def analyze_edge_characteristics(self):
        """Analyze edge computing characteristics in the dataset"""
        print("\n🔍 Analyzing Edge Computing Characteristics...")
        
        nodes = self.dataset_info['Nodes']
        edges = self.dataset_info['Edges']
        
        # Analyze node types
        device_types = {}
        for node in nodes:
            device_type = node.get('DeviceType', 'Unknown')
            if device_type not in device_types:
                device_types[device_type] = 0
            device_types[device_type] += 1
        
        print("📊 Node Distribution:")
        for device_type, count in device_types.items():
            print(f"   {device_type}: {count} nodes")
        
        # Analyze computational capabilities
        edge_nodes = [n for n in nodes if n.get('DeviceType') == 'Edge']
        fog_nodes = [n for n in nodes if n.get('DeviceType') == 'Fog' or 'f' in n.get('NodeName', '')]
        cloud_nodes = [n for n in nodes if n.get('DeviceType') == 'Cloud']
        
        print("\n💻 Computational Analysis:")
        if edge_nodes:
            edge_cpu = np.mean([n['MaxCpuFreq'] for n in edge_nodes])
            print(f"   Edge Avg CPU: {edge_cpu:,.0f} MHz")
        
        if fog_nodes:
            fog_cpu = np.mean([n['MaxCpuFreq'] for n in fog_nodes])
            print(f"   Fog Avg CPU: {fog_cpu:,.0f} MHz")
            
        if cloud_nodes:
            cloud_cpu = np.mean([n['MaxCpuFreq'] for n in cloud_nodes])
            print(f"   Cloud Avg CPU: {cloud_cpu:,.0f} MHz")
        
        # Save analysis
        analysis = {
            'device_distribution': device_types,
            'edge_nodes': len(edge_nodes),
            'fog_nodes': len(fog_nodes),
            'cloud_nodes': len(cloud_nodes),
            'total_nodes': len(nodes),
            'total_edges': len(edges)
        }
        
        return analysis
    
    def simulate_tasks_with_trust(self, tasks, num_tasks=1000):
        """Simulate tasks and collect trust-related features"""
        print(f"🔄 Simulating {min(num_tasks, len(tasks))} tasks...")
        
        features = []
        labels = []
        task_count = 0
        
        for task in tasks[:num_tasks]:
            try:
                # Set task parameters
                self.simulator.set_task(
                    cpu_demand=task.get('CpuDemand', 1000),
                    memory_demand=task.get('MemoryDemand', 512),
                    storage_demand=task.get('StorageDemand', 100),
                    deadline=task.get('Deadline', 10.0),
                    data_size=task.get('DataSize', 1024)
                )
                
                # Get available nodes
                available_nodes = list(range(len(self.dataset_info['Nodes'])))
                
                if not available_nodes:
                    continue
                
                # Simulate on a random node
                selected_node = np.random.choice(available_nodes)
                
                # Get node information
                node_info = self.dataset_info['Nodes'][selected_node]
                
                # Calculate basic metrics
                cpu_util = task.get('CpuDemand', 1000) / node_info['MaxCpuFreq']
                memory_util = task.get('MemoryDemand', 512) / node_info['MaxBufferSize']
                
                # Trust-related features
                feature_vector = [
                    cpu_util,                                    # CPU utilization
                    memory_util,                                 # Memory utilization
                    task.get('Deadline', 10.0),                # Deadline
                    task.get('DataSize', 1024),                 # Data size
                    node_info['MaxCpuFreq'],                    # Node capacity
                    node_info['MaxBufferSize'],                 # Node memory
                    node_info['IdleEnergyCoef'],                # Energy efficiency
                    selected_node,                               # Node ID
                    1 if node_info.get('DeviceType') == 'Edge' else 0,    # Is Edge
                    1 if node_info.get('DeviceType') == 'Fog' else 0,     # Is Fog
                    1 if node_info.get('DeviceType') == 'Cloud' else 0,   # Is Cloud
                ]
                
                # Trust label (simplified - based on resource adequacy)
                trust_score = 1.0 if (cpu_util < 0.8 and memory_util < 0.8) else 0.0
                
                features.append(feature_vector)
                labels.append(trust_score)
                task_count += 1
                
                if task_count % 500 == 0:
                    print(f"   Processed {task_count} tasks...")
                    
            except Exception as e:
                print(f"⚠️ Task simulation error: {e}")
                continue
        
        print(f"✅ Successfully simulated {len(features)} tasks")
        return np.array(features), np.array(labels)
    
    def create_edge_focused_models(self, input_dim):
        """Create GNN models optimized for edge computing scenarios"""
        print(f"🧠 Creating Edge-Optimized GNN Models (input_dim={input_dim})...")
        
        models = {}
        
        try:
            # GCN - Good for hierarchical edge-fog-cloud structures
            models['GCN'] = TrustGCN(
                input_dim=input_dim,
                hidden_dim=self.model_configs['GCN']['hidden_dim'],
                num_classes=2,
                num_layers=self.model_configs['GCN']['num_layers'],
                dropout=self.model_configs['GCN']['dropout']
            )
            
            # GAT - Excellent for attention-based trust relationships
            models['GAT'] = TrustGAT(
                input_dim=input_dim,
                hidden_dim=self.model_configs['GAT']['hidden_dim'],
                num_classes=2,
                num_heads=self.model_configs['GAT']['num_heads'],
                num_layers=self.model_configs['GAT']['num_layers'],
                dropout=self.model_configs['GAT']['dropout']
            )
            
            # GraphSAGE - Great for dynamic edge environments
            models['GraphSAGE'] = TrustGraphSAGE(
                input_dim=input_dim,
                hidden_dim=self.model_configs['GraphSAGE']['hidden_dim'],
                num_classes=2,
                num_layers=self.model_configs['GraphSAGE']['num_layers'],
                dropout=self.model_configs['GraphSAGE']['dropout']
            )
            
            # GIN - Powerful for complex edge topologies
            models['GIN'] = TrustGIN(
                input_dim=input_dim,
                hidden_dim=self.model_configs['GIN']['hidden_dim'],
                num_classes=2,
                num_layers=self.model_configs['GIN']['num_layers'],
                dropout=self.model_configs['GIN']['dropout']
            )
            
            print(f"✅ Created {len(models)} edge-optimized models")
            
        except Exception as e:
            print(f"❌ Error creating models: {e}")
            
        return models
    
    def create_graph_data(self, features, labels):
        """Create graph data structure for GNN training"""
        try:
            num_nodes = len(self.dataset_info['Nodes'])
            num_features = len(features)
            
            # Create adjacency matrix from dataset edges
            edge_list = []
            for edge in self.dataset_info['Edges']:
                src = edge['SrcNodeID']
                dst = edge['DstNodeID']
                edge_list.append([src, dst])
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # Create node features (aggregate task features per node)
            node_features = []
            for node_id in range(num_nodes):
                # Get features for tasks assigned to this node
                node_task_features = []
                for i, feature in enumerate(features):
                    if len(feature) > 7 and int(feature[7]) == node_id:  # Node ID is at index 7
                        node_task_features.append(feature[:7])  # Take first 7 features
                
                if node_task_features:
                    avg_features = np.mean(node_task_features, axis=0)
                else:
                    # Default features for nodes with no tasks
                    node_info = self.dataset_info['Nodes'][node_id]
                    avg_features = [
                        0.5,  # Default CPU util
                        0.5,  # Default memory util
                        10.0, # Default deadline
                        1024, # Default data size
                        node_info['MaxCpuFreq'],
                        node_info['MaxBufferSize'],
                        node_info['IdleEnergyCoef']
                    ]
                
                node_features.append(avg_features)
            
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Create labels for nodes (average trust of assigned tasks)
            node_labels = []
            for node_id in range(num_nodes):
                node_task_labels = []
                for i, feature in enumerate(features):
                    if len(feature) > 7 and int(feature[7]) == node_id:
                        node_task_labels.append(labels[i])
                
                if node_task_labels:
                    avg_label = np.mean(node_task_labels)
                else:
                    avg_label = 0.5  # Default neutral trust
                    
                node_labels.append(1 if avg_label > 0.5 else 0)
            
            y = torch.tensor(node_labels, dtype=torch.long)
            
            # Create data object
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index, y=y)
            
            print(f"📊 Graph Data Created:")
            print(f"   Nodes: {data.x.shape[0]}")
            print(f"   Edges: {data.edge_index.shape[1]}")
            print(f"   Features per node: {data.x.shape[1]}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error creating graph data: {e}")
            return None
    
    def train_edge_models(self, train_data, val_data, epochs=200):
        """Train GNN models with edge computing focus"""
        print(f"🏋️ Training Edge-Optimized Models ({epochs} epochs)...")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔧 Training {model_name}...")
            
            try:
                # Training setup
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                criterion = torch.nn.CrossEntropyLoss()
                
                model.train()
                train_losses = []
                val_accuracies = []
                
                best_val_acc = 0
                best_model_state = None
                
                for epoch in range(epochs):
                    # Training
                    optimizer.zero_grad()
                    out = model(train_data)
                    loss = criterion(out, train_data.y)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    
                    # Validation
                    if epoch % 20 == 0:
                        model.eval()
                        with torch.no_grad():
                            val_out = model(val_data)
                            val_pred = val_out.argmax(dim=1)
                            val_acc = accuracy_score(val_data.y.cpu(), val_pred.cpu())
                            val_accuracies.append(val_acc)
                            
                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_model_state = model.state_dict().copy()
                        
                        model.train()
                        print(f"   Epoch {epoch:3d}: Loss={loss.item():.4f}, Val_Acc={val_acc:.4f}")
                
                # Save best model
                if best_model_state:
                    model.load_state_dict(best_model_state)
                    torch.save(best_model_state, 
                             os.path.join(self.results_dir, f'best_{model_name.lower()}_edge_model.pth'))
                
                results[model_name] = {
                    'train_losses': train_losses,
                    'val_accuracies': val_accuracies,
                    'best_val_acc': best_val_acc,
                    'final_loss': train_losses[-1] if train_losses else 0
                }
                
                print(f"   ✅ {model_name} Best Validation Accuracy: {best_val_acc:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_edge_performance(self, test_data):
        """Evaluate models on edge computing scenarios"""
        print(f"\n📊 Evaluating Edge Computing Performance...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            try:
                model.eval()
                with torch.no_grad():
                    # Get predictions
                    out = model(test_data)
                    pred = out.argmax(dim=1)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(test_data.y.cpu(), pred.cpu())
                    
                    # Classification report
                    report = classification_report(
                        test_data.y.cpu(), 
                        pred.cpu(), 
                        target_names=['Untrusted', 'Trusted'],
                        output_dict=True
                    )
                    
                    evaluation_results[model_name] = {
                        'accuracy': accuracy,
                        'precision': report['Trusted']['precision'],
                        'recall': report['Trusted']['recall'],
                        'f1_score': report['Trusted']['f1-score'],
                        'report': report
                    }
                    
                    print(f"   Accuracy: {accuracy:.4f}")
                    print(f"   Precision: {report['Trusted']['precision']:.4f}")
                    print(f"   Recall: {report['Trusted']['recall']:.4f}")
                    print(f"   F1-Score: {report['Trusted']['f1-score']:.4f}")
                    
            except Exception as e:
                print(f"   ❌ Error evaluating {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def visualize_edge_results(self, training_results, evaluation_results):
        """Create comprehensive visualizations for edge computing results"""
        print(f"\n📈 Creating Edge Computing Visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Training Performance Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training losses
        for model_name, results in training_results.items():
            if 'train_losses' in results:
                ax1.plot(results['train_losses'], label=model_name, linewidth=2)
        ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation accuracies
        for model_name, results in training_results.items():
            if 'val_accuracies' in results:
                epochs = list(range(0, len(results['val_accuracies']) * 20, 20))
                ax2.plot(epochs, results['val_accuracies'], label=model_name, 
                        marker='o', linewidth=2, markersize=4)
        ax2.set_title('Validation Accuracy Progress', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Model performance comparison
        models = []
        accuracies = []
        f1_scores = []
        
        for model_name, results in evaluation_results.items():
            if 'accuracy' in results:
                models.append(model_name)
                accuracies.append(results['accuracy'])
                f1_scores.append(results['f1_score'])
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        ax3.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax3.bar(x_pos + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        ax3.set_title('Edge Computing Performance Metrics', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Precision-Recall comparison
        precisions = []
        recalls = []
        for model_name, results in evaluation_results.items():
            if 'precision' in results:
                precisions.append(results['precision'])
                recalls.append(results['recall'])
        
        if precisions and recalls:
            ax4.scatter(recalls, precisions, s=100, alpha=0.7)
            for i, model in enumerate(models):
                if i < len(precisions):
                    ax4.annotate(model, (recalls[i], precisions[i]), 
                               xytext=(5, 5), textcoords='offset points')
        
        ax4.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'edge_training_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Edge-specific analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance by model type
        model_types = list(evaluation_results.keys())
        performance_data = []
        
        for model in model_types:
            if model in evaluation_results and 'accuracy' in evaluation_results[model]:
                performance_data.append([
                    evaluation_results[model]['accuracy'],
                    evaluation_results[model]['precision'],
                    evaluation_results[model]['recall'],
                    evaluation_results[model]['f1_score']
                ])
        
        if performance_data:
            performance_df = pd.DataFrame(
                performance_data,
                columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                index=model_types
            )
            
            sns.heatmap(performance_df, annot=True, cmap='YlOrRd', ax=ax1, 
                       cbar_kws={'label': 'Score'})
            ax1.set_title('Edge Computing Performance Heatmap', fontsize=14, fontweight='bold')
        
        # Best model comparison
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x].get('f1_score', 0))
        
        if best_model in evaluation_results:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [
                evaluation_results[best_model]['accuracy'],
                evaluation_results[best_model]['precision'],
                evaluation_results[best_model]['recall'],
                evaluation_results[best_model]['f1_score']
            ]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
            ax2.pie(values, labels=metrics, autopct='%1.3f', colors=colors)
            ax2.set_title(f'Best Model Performance: {best_model}', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'edge_analysis_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to {self.results_dir}")
    
    def generate_edge_report(self, analysis, training_results, evaluation_results):
        """Generate comprehensive edge computing report"""
        print(f"\n📝 Generating Edge Computing Report...")
        
        report_path = os.path.join(self.results_dir, 'EDGE_COMPUTING_REPORT.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# 🔧 Edge Computing GNN Trust System Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Dataset:** {self.dataset_name}\n")
            f.write(f"**Focus:** Edge Computing Trust Management\n\n")
            
            # Dataset Analysis
            f.write(f"## 📊 Dataset Analysis\n\n")
            f.write(f"- **Total Nodes:** {analysis['total_nodes']}\n")
            f.write(f"- **Edge Nodes:** {analysis['edge_nodes']}\n")
            f.write(f"- **Fog Nodes:** {analysis['fog_nodes']}\n")
            f.write(f"- **Cloud Nodes:** {analysis['cloud_nodes']}\n")
            f.write(f"- **Total Edges:** {analysis['total_edges']}\n\n")
            
            f.write(f"### Device Distribution\n")
            for device_type, count in analysis['device_distribution'].items():
                f.write(f"- **{device_type}:** {count} nodes\n")
            f.write(f"\n")
            
            # Training Results
            f.write(f"## 🏋️ Training Results\n\n")
            f.write(f"| Model | Best Val Accuracy | Final Loss |\n")
            f.write(f"|-------|------------------|------------|\n")
            
            for model_name, results in training_results.items():
                if 'best_val_acc' in results:
                    f.write(f"| {model_name} | {results['best_val_acc']:.4f} | {results['final_loss']:.4f} |\n")
            f.write(f"\n")
            
            # Evaluation Results
            f.write(f"## 📊 Evaluation Results\n\n")
            f.write(f"| Model | Accuracy | Precision | Recall | F1-Score |\n")
            f.write(f"|-------|----------|-----------|--------|-----------|\n")
            
            for model_name, results in evaluation_results.items():
                if 'accuracy' in results:
                    f.write(f"| {model_name} | {results['accuracy']:.4f} | "
                           f"{results['precision']:.4f} | {results['recall']:.4f} | "
                           f"{results['f1_score']:.4f} |\n")
            f.write(f"\n")
            
            # Best Model
            if evaluation_results:
                best_model = max(evaluation_results.keys(), 
                               key=lambda x: evaluation_results[x].get('f1_score', 0))
                
                f.write(f"## 🏆 Best Performing Model\n\n")
                f.write(f"**{best_model}** achieved the highest F1-Score of "
                       f"**{evaluation_results[best_model]['f1_score']:.4f}**\n\n")
            
            # Edge Computing Insights
            f.write(f"## 🔍 Edge Computing Insights\n\n")
            f.write(f"1. **Hierarchical Architecture:** The Pakistan dataset provides an excellent "
                   f"representation of edge-fog-cloud hierarchies with {analysis['edge_nodes']} edge nodes, "
                   f"{analysis['fog_nodes']} fog nodes, and {analysis['cloud_nodes']} cloud nodes.\n\n")
            
            f.write(f"2. **Trust Dynamics:** Edge computing environments require dynamic trust assessment "
                   f"due to resource constraints and varying computational capabilities.\n\n")
            
            f.write(f"3. **Model Performance:** GNN models showed varying performance in edge scenarios, "
                   f"with attention-based models potentially performing better due to the need to focus "
                   f"on critical trust relationships.\n\n")
            
            # Recommendations
            f.write(f"## 💡 Recommendations\n\n")
            f.write(f"1. **Edge Optimization:** Focus on lightweight models for edge deployment\n")
            f.write(f"2. **Trust Propagation:** Implement hierarchical trust propagation from edge to cloud\n")
            f.write(f"3. **Dynamic Retraining:** Continuously retrain models based on edge performance data\n")
            f.write(f"4. **Resource Awareness:** Incorporate real-time resource availability in trust decisions\n\n")
            
            # Files Generated
            f.write(f"## 📁 Generated Files\n\n")
            f.write(f"- `edge_training_results.png` - Training performance visualizations\n")
            f.write(f"- `edge_analysis_results.png` - Edge-specific analysis results\n")
            f.write(f"- Model checkpoints: `best_*_edge_model.pth`\n")
            f.write(f"- This report: `EDGE_COMPUTING_REPORT.md`\n\n")
            
            f.write(f"---\n")
            f.write(f"*Report generated by Edge GNN Retraining System v1.0*\n")
        
        print(f"✅ Report saved to {report_path}")
        return report_path
    
    def run_complete_edge_analysis(self):
        """Run complete edge computing analysis"""
        print(f"🚀 Starting Complete Edge Computing GNN Analysis...")
        
        # Initialize system
        if not self.initialize_system():
            print("❌ Failed to initialize system")
            return
        
        # Analyze edge characteristics
        analysis = self.analyze_edge_characteristics()
        
        # Simulate training tasks
        print(f"\n🔄 Simulating Training Tasks...")
        train_features, train_labels = self.simulate_tasks_with_trust(
            self.train_tasks, num_tasks=2000
        )
        
        # Simulate test tasks
        print(f"\n🔄 Simulating Test Tasks...")
        test_features, test_labels = self.simulate_tasks_with_trust(
            self.test_tasks, num_tasks=800
        )
        
        # Create graph data
        print(f"\n📊 Creating Graph Data...")
        all_features = np.vstack([train_features, test_features])
        all_labels = np.hstack([train_labels, test_labels])
        
        graph_data = self.create_graph_data(all_features, all_labels)
        if graph_data is None:
            print("❌ Failed to create graph data")
            return
        
        # Split data
        num_nodes = graph_data.x.shape[0]
        num_train = int(0.6 * num_nodes)
        num_val = int(0.2 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:num_train] = True
        val_mask[num_train:num_train + num_val] = True
        test_mask[num_train + num_val:] = True
        
        train_data = graph_data
        train_data.train_mask = train_mask
        val_data = graph_data
        val_data.val_mask = val_mask
        test_data = graph_data
        test_data.test_mask = test_mask
        
        # Create models
        input_dim = graph_data.x.shape[1]
        self.models = self.create_edge_focused_models(input_dim)
        
        # Train models
        training_results = self.train_edge_models(train_data, val_data, epochs=200)
        
        # Evaluate models
        evaluation_results = self.evaluate_edge_performance(test_data)
        
        # Create visualizations
        self.visualize_edge_results(training_results, evaluation_results)
        
        # Generate report
        report_path = self.generate_edge_report(analysis, training_results, evaluation_results)
        
        print(f"\n🎉 Edge Computing Analysis Complete!")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"📄 Report available at: {report_path}")
        
        return {
            'analysis': analysis,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'results_dir': self.results_dir,
            'report_path': report_path
        }

def main():
    """Main execution function"""
    print("🔧 Edge Computing GNN Trust System")
    print("=" * 50)
    
    try:
        # Create system
        system = EdgeGNNRetrainingSystem()
        
        # Run complete analysis
        results = system.run_complete_edge_analysis()
        
        if results:
            print("\n✅ Edge Computing Analysis Successfully Completed!")
            print(f"Check {results['results_dir']} for detailed results.")
        else:
            print("\n❌ Analysis failed!")
    
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()