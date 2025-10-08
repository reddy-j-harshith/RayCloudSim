#!/usr/bin/env python3
"""
GNN Trust Value Regression System - Fixed Implementation

This system:
1. Uses real topologies (pakistan/topo4mec) instead of synthetic data
2. Treats trust as continuous values (regression) not binary classification
3. Uses threshold-based malicious node detection from trust values
4. Properly trains on real network data for accurate attack detection

Key fixes:
- Trust prediction is regression outputting continuous values [0,1]
- Malicious node detection uses trust thresholds (static or statistical)
- Real topology loading from JSON configs
- MSE loss for trust regression instead of classification accuracy
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, TransformerConv, global_mean_pool
import networkx as nx
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TrustRegressionDataset:
    """Dataset for trust value regression using real topologies"""
    
    def __init__(self, config_paths, num_samples=1000):
        self.config_paths = config_paths
        self.num_samples = num_samples
        self.graphs = []
        self.trust_values = []
        self.malicious_labels = []
        
        print(f"Loading real topologies from {len(config_paths)} configurations...")
        self._load_real_topologies()
        self._generate_trust_regression_data()
        
    def _load_real_topologies(self):
        """Load real network topologies from JSON configs"""
        self.topologies = []
        
        for config_path in self.config_paths:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Build NetworkX graph from config
                G = nx.DiGraph()
                
                # Add nodes with attributes
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
                
                # Add edges
                for edge in config['Edges']:
                    src = edge['SrcNodeID']
                    dst = edge['DstNodeID']
                    bandwidth = edge.get('Bandwidth', 1000.0)
                    G.add_edge(src, dst, bandwidth=bandwidth)
                
                self.topologies.append({
                    'graph': G,
                    'node_types': node_types,
                    'config_name': os.path.basename(config_path)
                })
                
                print(f"Loaded topology {os.path.basename(config_path)}: "
                      f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                
            except Exception as e:
                print(f"Error loading {config_path}: {e}")
                continue
    
    def _compute_node_features(self, G, node_id):
        """Compute comprehensive node features"""
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
        node_type = node_data.get('node_type', 'TrustNode')
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
    
    def _generate_trust_values(self, G, node_types):
        """Generate realistic trust values based on node behavior and network position"""
        trust_values = {}
        
        for node_id in G.nodes():
            node_type = node_types.get(node_id, 'TrustNode')
            
            # Base trust based on node type
            if node_type == 'MaliciousNode':
                # Malicious nodes have low trust (0.1 to 0.4)
                base_trust = np.random.uniform(0.1, 0.4)
            elif node_type in ['TrustNode', 'GNNTrustNode']:
                # Trust nodes have high trust (0.6 to 0.95)
                base_trust = np.random.uniform(0.6, 0.95)
            else:
                # Other nodes have medium trust (0.4 to 0.8)
                base_trust = np.random.uniform(0.4, 0.8)
            
            # Modify trust based on network position
            try:
                degree_cent = nx.degree_centrality(G)[node_id]
                betweenness_cent = nx.betweenness_centrality(G).get(node_id, 0.0)
                
                # Higher centrality can increase trust for non-malicious nodes
                if node_type != 'MaliciousNode':
                    centrality_bonus = (degree_cent + betweenness_cent) * 0.1
                    base_trust = min(0.99, base_trust + centrality_bonus)
                else:
                    # For malicious nodes, higher centrality makes them more dangerous
                    centrality_penalty = (degree_cent + betweenness_cent) * 0.05
                    base_trust = max(0.01, base_trust - centrality_penalty)
            except:
                pass
            
            # Add some noise
            noise = np.random.normal(0, 0.05)
            final_trust = np.clip(base_trust + noise, 0.01, 0.99)
            
            trust_values[node_id] = final_trust
        
        return trust_values
    
    def _generate_trust_regression_data(self):
        """Generate training data for trust regression"""
        print("Generating trust regression training data...")
        
        for sample_idx in range(self.num_samples):
            # Select random topology
            topo = np.random.choice(self.topologies)
            G = topo['graph']
            node_types = topo['node_types']
            
            # Generate trust values for this sample
            trust_values = self._generate_trust_values(G, node_types)
            
            # Convert to PyTorch Geometric format
            node_features = []
            node_ids = list(G.nodes())
            
            for node_id in node_ids:
                features = self._compute_node_features(G, node_id)
                node_features.append(features)
            
            # Edge indices
            edge_index = []
            for src, dst in G.edges():
                src_idx = node_ids.index(src)
                dst_idx = node_ids.index(dst)
                edge_index.append([src_idx, dst_idx])
            
            if len(edge_index) == 0:
                continue
            
            # Create PyTorch Geometric data
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Trust values as regression targets
            trust_targets = torch.tensor([trust_values[node_id] for node_id in node_ids], 
                                       dtype=torch.float)
            
            # Malicious labels based on node types (for evaluation)
            malicious_labels = torch.tensor([1 if node_types.get(node_id) == 'MaliciousNode' else 0 
                                           for node_id in node_ids], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=trust_targets, 
                       malicious_labels=malicious_labels)
            
            self.graphs.append(data)
            
            if (sample_idx + 1) % 100 == 0:
                print(f"Generated {sample_idx + 1}/{self.num_samples} samples")
        
        print(f"Total samples generated: {len(self.graphs)}")

class GNNTrustRegressor(nn.Module):
    """GNN model for trust value regression"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, model_type='GAT'):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        if model_type == 'GAT':
            heads = 4
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=0.1))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.1))
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.1))
            
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                
        elif model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                
        elif model_type == 'Transformer':
            self.convs.append(TransformerConv(input_dim, hidden_dim, dropout=0.1))
            for _ in range(num_layers - 1):
                self.convs.append(TransformerConv(hidden_dim, hidden_dim, dropout=0.1))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output trust values in [0, 1]
        )
        
    def forward(self, x, edge_index, batch=None):
        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Regression output
        trust_values = self.regression_head(x).squeeze()
        
        return trust_values

class TrustRegressionTrainer:
    """Trainer for GNN trust regression"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_trust = self.model(data.x, data.edge_index, data.batch)
            
            # Regression loss
            loss = self.criterion(pred_trust, data.y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * data.y.size(0)
            num_samples += data.y.size(0)
        
        return total_loss / num_samples
    
    def evaluate(self, val_loader, trust_threshold=0.5):
        """Evaluate model with trust regression metrics and malicious detection"""
        self.model.eval()
        total_loss = 0
        num_samples = 0
        
        all_pred_trust = []
        all_true_trust = []
        all_pred_malicious = []
        all_true_malicious = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                
                # Predict trust values
                pred_trust = self.model(data.x, data.edge_index, data.batch)
                
                # Regression loss
                loss = self.criterion(pred_trust, data.y)
                total_loss += loss.item() * data.y.size(0)
                num_samples += data.y.size(0)
                
                # Collect predictions and ground truth
                all_pred_trust.extend(pred_trust.cpu().numpy())
                all_true_trust.extend(data.y.cpu().numpy())
                
                # Malicious detection using trust threshold
                pred_malicious = (pred_trust < trust_threshold).long()
                all_pred_malicious.extend(pred_malicious.cpu().numpy())
                all_true_malicious.extend(data.malicious_labels.cpu().numpy())
        
        # Regression metrics
        mse = mean_squared_error(all_true_trust, all_pred_trust)
        mae = mean_absolute_error(all_true_trust, all_pred_trust)
        rmse = np.sqrt(mse)
        
        # Malicious detection metrics
        true_pos = sum(1 for t, p in zip(all_true_malicious, all_pred_malicious) if t == 1 and p == 1)
        false_pos = sum(1 for t, p in zip(all_true_malicious, all_pred_malicious) if t == 0 and p == 1)
        true_neg = sum(1 for t, p in zip(all_true_malicious, all_pred_malicious) if t == 0 and p == 0)
        false_neg = sum(1 for t, p in zip(all_true_malicious, all_pred_malicious) if t == 1 and p == 0)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (true_pos + true_neg) / len(all_true_malicious)
        
        return {
            'loss': total_loss / num_samples,
            'mse': mse,
            'mae': mae, 
            'rmse': rmse,
            'detection_accuracy': accuracy,
            'detection_precision': precision,
            'detection_recall': recall,
            'detection_f1': f1
        }

def detect_malicious_nodes_statistical(trust_values, method='zscore', threshold=2.0):
    """Statistical methods for malicious node detection"""
    trust_array = np.array(trust_values)
    
    if method == 'zscore':
        # Z-score based detection
        mean_trust = np.mean(trust_array)
        std_trust = np.std(trust_array)
        z_scores = np.abs((trust_array - mean_trust) / std_trust)
        malicious_mask = (trust_array < mean_trust) & (z_scores > threshold)
        
    elif method == 'iqr':
        # Interquartile range based detection
        q1 = np.percentile(trust_array, 25)
        q3 = np.percentile(trust_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        malicious_mask = trust_array < lower_bound
        
    elif method == 'percentile':
        # Bottom percentile as malicious
        percentile_threshold = np.percentile(trust_array, threshold)
        malicious_mask = trust_array <= percentile_threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return malicious_mask

def main():
    """Main training function"""
    print("GNN Trust Value Regression Training")
    print("=" * 50)
    
    # Configuration paths for real topologies
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
        print("No topology configurations found! Please check paths.")
        return
    
    # Create dataset
    dataset = TrustRegressionDataset(existing_configs, num_samples=2000)
    
    if not dataset.graphs:
        print("No training data generated! Check topology loading.")
        return
    
    # Split dataset
    train_data, test_data = train_test_split(dataset.graphs, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    print(f"Dataset splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Get input dimension
    input_dim = dataset.graphs[0].x.shape[1]
    print(f"Input feature dimension: {input_dim}")
    
    # Train different model types
    model_types = ['GAT', 'GraphSAGE', 'GCN', 'Transformer']
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        
        # Create model
        model = GNNTrustRegressor(input_dim, hidden_dim=128, model_type=model_type)
        trainer = TrustRegressionTrainer(model)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(100):
            # Train
            train_loss = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            # Learning rate scheduling
            trainer.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_{model_type.lower()}_trust_regressor.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val RMSE={val_metrics['rmse']:.4f}, "
                      f"Detection Acc={val_metrics['detection_accuracy']:.4f}")
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and test
        model.load_state_dict(torch.load(f'best_{model_type.lower()}_trust_regressor.pth'))
        test_metrics = trainer.evaluate(test_loader)
        results[model_type] = test_metrics
        
        print(f"\n{model_type} Test Results:")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Malicious Detection Accuracy: {test_metrics['detection_accuracy']:.4f}")
        print(f"  Malicious Detection Precision: {test_metrics['detection_precision']:.4f}")
        print(f"  Malicious Detection Recall: {test_metrics['detection_recall']:.4f}")
        print(f"  Malicious Detection F1: {test_metrics['detection_f1']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'gnn_trust_regression_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    print("\nFinal Results Summary:")
    print(results_df)
    
    # Demonstrate statistical threshold methods
    print("\nTesting Statistical Threshold Methods:")
    
    # Get sample trust predictions for demonstration
    model.eval()
    sample_data = test_data[0]
    with torch.no_grad():
        sample_trust = model(sample_data.x, sample_data.edge_index).cpu().numpy()
    
    methods = [('zscore', 2.0), ('iqr', None), ('percentile', 20)]
    for method, threshold in methods:
        if threshold is not None:
            malicious_detected = detect_malicious_nodes_statistical(sample_trust, method, threshold)
        else:
            malicious_detected = detect_malicious_nodes_statistical(sample_trust, method)
        
        print(f"  {method}: Detected {np.sum(malicious_detected)} malicious nodes")

if __name__ == "__main__":
    main()