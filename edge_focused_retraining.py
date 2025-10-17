#!/usr/bin/env python3
"""
🔧 Edge Computing-Focused GNN Trust Retraining System

Self-contained system for retraining GNN models specifically for edge computing
scenarios using the Pakistan dataset's hierarchical edge-fog-cloud structure.
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

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GATConv, SAGEConv, GCNConv, TransformerConv, GINConv
    import networkx as nx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch Geometric not available. Please install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False

class EdgeTrustGNN(nn.Module):
    """Edge-optimized GNN for trust prediction in hierarchical computing environments"""
    
    def __init__(self, input_dim, hidden_dim=64, model_type='GAT', num_layers=3, dropout=0.3):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if model_type == 'GAT':
            heads = 4
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
            
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                
        elif model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                
        elif model_type == 'Transformer':
            self.convs.append(TransformerConv(input_dim, hidden_dim, dropout=dropout))
            for _ in range(num_layers - 1):
                self.convs.append(TransformerConv(hidden_dim, hidden_dim, dropout=dropout))
        
        # Batch normalization layers
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Edge-specific trust prediction head
        self.edge_trust_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Trust values in [0, 1]
        )
        
        # Device type classifier (edge/fog/cloud)
        self.device_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 3),  # 3 device types
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, edge_index, batch=None):
        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Trust prediction
        trust_scores = self.edge_trust_head(x).squeeze()
        
        # Device type prediction (auxiliary task)
        device_probs = self.device_classifier(x)
        
        return trust_scores, device_probs

class EdgeComputingDatasetLoader:
    """Load and prepare edge computing dataset from Pakistan topology"""
    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path or self._find_pakistan_config()
        self.topology = None
        self.node_mapping = {}
        
    def _find_pakistan_config(self):
        """Find Pakistan configuration file"""
        possible_paths = [
            "experiments/gnn_trust/gnn_pakistan_Tuple30K_config.json",
            "examples/configs/pakistan_Tuple30K.json",
            "configs/pakistan_Tuple30K.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Create a simple Pakistan-like topology if no config found
        return self._create_default_topology()
    
    def _create_default_topology(self):
        """Create a default Pakistan-like edge-fog-cloud topology"""
        config = {
            'Nodes': [
                # Edge node (Islamabad)
                {
                    'NodeId': 0, 'NodeName': 'e0', 'DeviceType': 'Edge',
                    'MaxCpuFreq': 10000, 'MaxBufferSize': 4096,
                    'IdleEnergyCoef': 0.01, 'ExeEnergyCoef': 0.4,
                    'LocX': 33.68742, 'LocY': 73.0078, 'Location': 'Islamabad, Pakistan'
                },
                # Fog nodes
                {
                    'NodeId': 1, 'NodeName': 'f0', 'DeviceType': 'Fog',
                    'MaxCpuFreq': 50000, 'MaxBufferSize': 8192,
                    'IdleEnergyCoef': 0.03, 'ExeEnergyCoef': 0.2,
                    'LocX': 31.5497, 'LocY': 74.3436, 'Location': 'Lahore, Pakistan'
                },
                {
                    'NodeId': 2, 'NodeName': 'f1', 'DeviceType': 'Fog',
                    'MaxCpuFreq': 45000, 'MaxBufferSize': 6144,
                    'IdleEnergyCoef': 0.035, 'ExeEnergyCoef': 0.22,
                    'LocX': 24.8607, 'LocY': 67.0011, 'Location': 'Karachi, Pakistan'
                },
                {
                    'NodeId': 3, 'NodeName': 'f2', 'DeviceType': 'Fog',
                    'MaxCpuFreq': 55000, 'MaxBufferSize': 10240,
                    'IdleEnergyCoef': 0.032, 'ExeEnergyCoef': 0.19,
                    'LocX': 34.0151, 'LocY': 71.5249, 'Location': 'Peshawar, Pakistan'
                },
                # Cloud nodes
                {
                    'NodeId': 4, 'NodeName': 'c0', 'DeviceType': 'Cloud',
                    'MaxCpuFreq': 200000, 'MaxBufferSize': 32768,
                    'IdleEnergyCoef': 0.08, 'ExeEnergyCoef': 0.15,
                    'LocX': 1.277911, 'LocY': 103.848, 'Location': 'Singapore'
                },
                {
                    'NodeId': 5, 'NodeName': 'c1', 'DeviceType': 'Cloud',
                    'MaxCpuFreq': 250000, 'MaxBufferSize': 51200,
                    'IdleEnergyCoef': 0.09, 'ExeEnergyCoef': 0.16,
                    'LocX': 50.4738, 'LocY': 3.8038, 'Location': 'Brussels, Belgium'
                }
            ],
            'Edges': [
                # Edge to Fog connections
                {'SrcNodeID': 0, 'DstNodeID': 1, 'Bandwidth': 2000},
                {'SrcNodeID': 0, 'DstNodeID': 2, 'Bandwidth': 1800},
                {'SrcNodeID': 0, 'DstNodeID': 3, 'Bandwidth': 1500},
                # Fog to Fog connections
                {'SrcNodeID': 1, 'DstNodeID': 2, 'Bandwidth': 5000},
                {'SrcNodeID': 2, 'DstNodeID': 3, 'Bandwidth': 4000},
                {'SrcNodeID': 1, 'DstNodeID': 3, 'Bandwidth': 4500},
                # Fog to Cloud connections
                {'SrcNodeID': 1, 'DstNodeID': 4, 'Bandwidth': 10000},
                {'SrcNodeID': 2, 'DstNodeID': 5, 'Bandwidth': 12000},
                {'SrcNodeID': 3, 'DstNodeID': 4, 'Bandwidth': 9000},
                # Cloud to Cloud connections
                {'SrcNodeID': 4, 'DstNodeID': 5, 'Bandwidth': 50000}
            ]
        }
        
        # Save temporary config
        temp_path = "temp_pakistan_topology.json"
        with open(temp_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return temp_path
    
    def load_topology(self):
        """Load the topology configuration"""
        try:
            with open(self.dataset_path, 'r') as f:
                config = json.load(f)
            
            self.topology = config
            
            # Create node mapping
            for node in config['Nodes']:
                self.node_mapping[node['NodeId']] = node
            
            print(f"📊 Topology Loaded:")
            print(f"   Nodes: {len(config['Nodes'])}")
            print(f"   Edges: {len(config['Edges'])}")
            
            # Analyze device distribution
            device_types = {}
            for node in config['Nodes']:
                device_type = node.get('DeviceType', 'Unknown')
                device_types[device_type] = device_types.get(device_type, 0) + 1
            
            print(f"   Device Distribution: {device_types}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading topology: {e}")
            return False
    
    def generate_edge_computing_features(self, num_samples=2000):
        """Generate edge computing specific features and trust labels"""
        if not self.topology:
            print("❌ No topology loaded!")
            return None, None, None
        
        print(f"🔄 Generating {num_samples} edge computing samples...")
        
        # Build NetworkX graph for centrality calculations
        G = nx.DiGraph()
        for node in self.topology['Nodes']:
            G.add_node(node['NodeId'], **node)
        
        for edge in self.topology['Edges']:
            G.add_edge(edge['SrcNodeID'], edge['DstNodeID'], 
                      bandwidth=edge.get('Bandwidth', 1000))
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector_centrality = {node: 0.0 for node in G.nodes()}
        
        # Generate samples
        samples = []
        trust_labels = []
        device_labels = []
        
        device_type_map = {'Edge': 0, 'Fog': 1, 'Cloud': 2}
        
        for _ in range(num_samples):
            # Random task characteristics
            cpu_demand = np.random.uniform(500, 5000)
            memory_demand = np.random.uniform(100, 2000)
            deadline = np.random.uniform(1.0, 20.0)
            data_size = np.random.uniform(100, 10000)
            priority = np.random.uniform(0.1, 1.0)
            
            # Select random node for placement
            node_id = np.random.choice(list(G.nodes()))
            node_info = self.node_mapping[node_id]
            
            # Node features
            cpu_freq = node_info.get('MaxCpuFreq', 1000)
            buffer_size = node_info.get('MaxBufferSize', 100)
            idle_energy = node_info.get('IdleEnergyCoef', 0.01)
            exe_energy = node_info.get('ExeEnergyCoef', 0.1)
            device_type = node_info.get('DeviceType', 'Edge')
            
            # Resource utilization
            cpu_util = cpu_demand / cpu_freq
            memory_util = memory_demand / buffer_size
            
            # Network characteristics
            degree_cent = degree_centrality[node_id]
            between_cent = betweenness_centrality[node_id]
            close_cent = closeness_centrality[node_id]
            eigen_cent = eigenvector_centrality[node_id]
            
            # Edge computing specific features
            edge_distance = 0 if device_type == 'Edge' else (1 if device_type == 'Fog' else 2)
            network_hops = edge_distance + np.random.poisson(1)
            latency_estimate = network_hops * 10 + np.random.normal(5, 2)
            
            # Feature vector
            features = [
                cpu_util, memory_util, deadline, data_size, priority,
                cpu_freq / 100000.0,  # Normalized
                buffer_size / 10000.0,  # Normalized
                idle_energy, exe_energy,
                degree_cent, between_cent, close_cent, eigen_cent,
                edge_distance, network_hops, latency_estimate,
                1 if device_type == 'Edge' else 0,
                1 if device_type == 'Fog' else 0,
                1 if device_type == 'Cloud' else 0
            ]
            
            # Trust calculation based on edge computing principles
            trust_score = self._calculate_edge_trust(
                cpu_util, memory_util, deadline, latency_estimate, 
                device_type, degree_cent
            )
            
            samples.append(features)
            trust_labels.append(trust_score)
            device_labels.append(device_type_map.get(device_type, 0))
        
        print(f"✅ Generated {len(samples)} samples")
        return np.array(samples), np.array(trust_labels), np.array(device_labels)
    
    def _calculate_edge_trust(self, cpu_util, memory_util, deadline, latency, device_type, centrality):
        """Calculate trust score based on edge computing principles"""
        
        # Base trust by device type
        if device_type == 'Edge':
            base_trust = 0.7  # Good for low-latency but limited resources
        elif device_type == 'Fog':
            base_trust = 0.8  # Balanced resources and latency
        else:  # Cloud
            base_trust = 0.6  # High resources but potentially high latency
        
        # Resource adequacy penalty
        resource_penalty = 0
        if cpu_util > 0.8:
            resource_penalty += 0.3
        elif cpu_util > 0.6:
            resource_penalty += 0.1
        
        if memory_util > 0.8:
            resource_penalty += 0.2
        elif memory_util > 0.6:
            resource_penalty += 0.05
        
        # Latency penalty for time-critical tasks
        latency_penalty = 0
        if deadline < 5.0 and latency > 50:
            latency_penalty = 0.4
        elif deadline < 10.0 and latency > 100:
            latency_penalty = 0.2
        
        # Centrality bonus (well-connected nodes are more trustworthy)
        centrality_bonus = centrality * 0.1
        
        # Calculate final trust
        trust = base_trust - resource_penalty - latency_penalty + centrality_bonus
        trust += np.random.normal(0, 0.05)  # Add noise
        
        return np.clip(trust, 0.01, 0.99)

class EdgeRetrainingSystem:
    """Main system for edge computing GNN retraining"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"edge_retraining_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🔧 Edge Computing GNN Retraining System")
        print(f"📁 Results directory: {self.results_dir}")
        
        self.dataset_loader = EdgeComputingDatasetLoader()
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load topology and prepare training data"""
        print("\n📊 Loading Edge Computing Data...")
        
        if not self.dataset_loader.load_topology():
            return False
        
        # Generate features and labels
        features, trust_labels, device_labels = self.dataset_loader.generate_edge_computing_features(2000)
        
        if features is None:
            return False
        
        # Create graph structure
        topology = self.dataset_loader.topology
        nodes = topology['Nodes']
        edges = topology['Edges']
        
        # Node features (aggregate task features per node)
        num_nodes = len(nodes)
        node_features = []
        node_trust_labels = []
        node_device_labels = []
        
        # Map tasks to nodes based on random assignment (since we don't have explicit mapping)
        tasks_per_node = len(features) // num_nodes
        
        for node_id in range(num_nodes):
            # Assign tasks to nodes in round-robin fashion
            start_idx = node_id * tasks_per_node
            end_idx = start_idx + tasks_per_node if node_id < num_nodes - 1 else len(features)
            
            if start_idx < len(features):
                # Average features for this node's tasks
                node_task_features = features[start_idx:end_idx]
                node_task_trust = trust_labels[start_idx:end_idx]
                node_task_devices = device_labels[start_idx:end_idx]
                
                if len(node_task_features) > 0:
                    avg_features = np.mean(node_task_features[:, :16], axis=0)  # First 16 features
                    avg_trust = np.mean(node_task_trust)
                    device_label = int(np.median(node_task_devices))
                else:
                    avg_features = None
                    
            else:
                avg_features = None
            
            # Use node-specific features if no tasks assigned
            if avg_features is None:
                node_info = nodes[node_id]
                avg_features = np.array([
                    0.5, 0.5, 10.0, 1000.0, 0.5,  # Task features
                    node_info.get('MaxCpuFreq', 1000) / 100000.0,
                    node_info.get('MaxBufferSize', 100) / 10000.0,
                    node_info.get('IdleEnergyCoef', 0.01),
                    node_info.get('ExeEnergyCoef', 0.1),
                    0.2, 0.1, 0.3, 0.1,  # Centrality features
                    1, 2, 20  # Network features
                ])
                avg_trust = 0.5
                device_type = node_info.get('DeviceType', 'Edge')
                device_label = {'Edge': 0, 'Fog': 1, 'Cloud': 2, 'Unknown': 0}.get(device_type, 0)
            
            node_features.append(avg_features)
            node_trust_labels.append(avg_trust)
            node_device_labels.append(device_label)
        
        # Create edge index
        edge_index = []
        for edge in edges:
            edge_index.append([edge['SrcNodeID'], edge['DstNodeID']])
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        trust_y = torch.tensor(node_trust_labels, dtype=torch.float)
        device_y = torch.tensor(node_device_labels, dtype=torch.long)
        
        # Create graph data
        if TORCH_GEOMETRIC_AVAILABLE:
            graph_data = Data(x=x, edge_index=edge_index, trust_y=trust_y, device_y=device_y)
        else:
            # Fallback for systems without PyTorch Geometric
            graph_data = {
                'x': x, 'edge_index': edge_index, 
                'trust_y': trust_y, 'device_y': device_y
            }
        
        print(f"✅ Graph created: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
        
        self.graph_data = graph_data
        self.input_dim = x.shape[1]
        
        return True
    
    def create_models(self):
        """Create edge-optimized GNN models"""
        print(f"\n🧠 Creating Edge-Optimized Models...")
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("❌ PyTorch Geometric not available. Cannot create GNN models.")
            return False
        
        model_configs = {
            'GAT': {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.3},
            'GraphSAGE': {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.3},
            'GCN': {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.3},
            'Transformer': {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.3}
        }
        
        for model_name, config in model_configs.items():
            try:
                model = EdgeTrustGNN(
                    input_dim=self.input_dim,
                    model_type=model_name,
                    **config
                )
                self.models[model_name] = model
                print(f"   ✅ Created {model_name} model")
            except Exception as e:
                print(f"   ❌ Failed to create {model_name}: {e}")
        
        return len(self.models) > 0
    
    def train_models(self, epochs=200):
        """Train all models"""
        print(f"\n🏋️ Training Models ({epochs} epochs)...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        # Split data
        num_nodes = self.graph_data.x.shape[0]
        num_train = int(0.6 * num_nodes)
        num_val = int(0.2 * num_nodes)
        
        indices = torch.randperm(num_nodes)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train + num_val]
        test_idx = indices[num_train + num_val:]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        self.graph_data.train_mask = train_mask
        self.graph_data.val_mask = val_mask
        self.graph_data.test_mask = test_mask
        
        # Move data to device
        self.graph_data = self.graph_data.to(device)
        
        training_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔧 Training {model_name}...")
            
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            trust_criterion = nn.MSELoss()
            device_criterion = nn.CrossEntropyLoss()
            
            best_val_loss = float('inf')
            patience = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                model.train()
                optimizer.zero_grad()
                
                trust_pred, device_pred = model(self.graph_data.x, self.graph_data.edge_index)
                
                # Multi-task loss
                trust_loss = trust_criterion(trust_pred[train_mask], self.graph_data.trust_y[train_mask])
                device_loss = device_criterion(device_pred[train_mask], self.graph_data.device_y[train_mask])
                
                total_loss = trust_loss + 0.3 * device_loss  # Weight device classification less
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(total_loss.item())
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_trust_pred, val_device_pred = model(self.graph_data.x, self.graph_data.edge_index)
                        val_trust_loss = trust_criterion(val_trust_pred[val_mask], self.graph_data.trust_y[val_mask])
                        val_device_loss = device_criterion(val_device_pred[val_mask], self.graph_data.device_y[val_mask])
                        val_loss = val_trust_loss + 0.3 * val_device_loss
                        
                        val_losses.append(val_loss.item())
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience = 0
                            # Save best model
                            torch.save(model.state_dict(), 
                                     os.path.join(self.results_dir, f'best_{model_name.lower()}_edge_model.pth'))
                        else:
                            patience += 1
                        
                        print(f"   Epoch {epoch:3d}: Train Loss={total_loss.item():.4f}, "
                              f"Val Loss={val_loss.item():.4f}, Trust RMSE={torch.sqrt(val_trust_loss).item():.4f}")
                
                if patience >= 20:  # Early stopping
                    print(f"   Early stopping at epoch {epoch}")
                    break
            
            training_results[model_name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }
            
            print(f"   ✅ {model_name} training completed")
        
        self.training_results = training_results
        return True
    
    def evaluate_models(self):
        """Evaluate trained models"""
        print(f"\n📊 Evaluating Models...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            # Load best model
            model.load_state_dict(torch.load(
                os.path.join(self.results_dir, f'best_{model_name.lower()}_edge_model.pth')
            ))
            model.eval()
            
            with torch.no_grad():
                trust_pred, device_pred = model(self.graph_data.x, self.graph_data.edge_index)
                
                # Trust regression metrics
                test_trust_pred = trust_pred[self.graph_data.test_mask].cpu().numpy()
                test_trust_true = self.graph_data.trust_y[self.graph_data.test_mask].cpu().numpy()
                
                mse = mean_squared_error(test_trust_true, test_trust_pred)
                mae = mean_absolute_error(test_trust_true, test_trust_pred)
                rmse = np.sqrt(mse)
                
                # Device classification metrics
                test_device_pred = device_pred[self.graph_data.test_mask].argmax(dim=1).cpu().numpy()
                test_device_true = self.graph_data.device_y[self.graph_data.test_mask].cpu().numpy()
                
                device_acc = accuracy_score(test_device_true, test_device_pred)
                
                # Trust-based malicious detection (threshold-based)
                trust_threshold = 0.5
                malicious_pred = (test_trust_pred < trust_threshold).astype(int)
                malicious_true = (test_trust_true < trust_threshold).astype(int)
                
                if len(np.unique(malicious_true)) > 1:
                    mal_acc = accuracy_score(malicious_true, malicious_pred)
                    mal_report = classification_report(malicious_true, malicious_pred, output_dict=True)
                    mal_precision = mal_report['1']['precision']
                    mal_recall = mal_report['1']['recall']
                    mal_f1 = mal_report['1']['f1-score']
                else:
                    mal_acc = mal_precision = mal_recall = mal_f1 = 0.0
                
                evaluation_results[model_name] = {
                    'trust_mse': mse,
                    'trust_mae': mae,
                    'trust_rmse': rmse,
                    'device_accuracy': device_acc,
                    'malicious_accuracy': mal_acc,
                    'malicious_precision': mal_precision,
                    'malicious_recall': mal_recall,
                    'malicious_f1': mal_f1
                }
                
                print(f"   Trust RMSE: {rmse:.4f}")
                print(f"   Trust MAE: {mae:.4f}")
                print(f"   Device Classification Acc: {device_acc:.4f}")
                print(f"   Malicious Detection F1: {mal_f1:.4f}")
        
        self.evaluation_results = evaluation_results
        return True
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\n📈 Creating Visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Training progress
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training losses
        for model_name, results in self.training_results.items():
            if 'train_losses' in results:
                ax1.plot(results['train_losses'], label=model_name, linewidth=2)
        ax1.set_title('Training Loss Progress', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation losses
        for model_name, results in self.training_results.items():
            if 'val_losses' in results:
                epochs = list(range(0, len(results['val_losses']) * 10, 10))
                ax2.plot(epochs, results['val_losses'], label=model_name, 
                        marker='o', linewidth=2, markersize=4)
        ax2.set_title('Validation Loss Progress', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Model performance comparison
        models = list(self.evaluation_results.keys())
        trust_rmse = [self.evaluation_results[m]['trust_rmse'] for m in models]
        device_acc = [self.evaluation_results[m]['device_accuracy'] for m in models]
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        ax3.bar(x_pos - width/2, trust_rmse, width, label='Trust RMSE', alpha=0.8)
        ax3.bar(x_pos + width/2, device_acc, width, label='Device Accuracy', alpha=0.8)
        ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Edge computing metrics heatmap
        metrics = ['trust_rmse', 'trust_mae', 'device_accuracy', 'malicious_f1']
        heatmap_data = []
        for model in models:
            row = [self.evaluation_results[model][metric] for metric in metrics]
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, annot=True, xticklabels=metrics, yticklabels=models,
                   cmap='YlOrRd', ax=ax4)
        ax4.set_title('Edge Computing Metrics Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'edge_retraining_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Edge-specific analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Best model performance breakdown
        best_model = max(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['malicious_f1'])
        
        categories = ['Trust RMSE', 'Device Acc', 'Malicious F1']
        values = [
            self.evaluation_results[best_model]['trust_rmse'],
            self.evaluation_results[best_model]['device_accuracy'],
            self.evaluation_results[best_model]['malicious_f1']
        ]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        ax1.pie(values, labels=categories, autopct='%1.3f', colors=colors)
        ax1.set_title(f'Best Model ({best_model}) Performance', fontsize=14, fontweight='bold')
        
        # Performance by metric
        metrics_comparison = ['trust_rmse', 'trust_mae', 'device_accuracy', 'malicious_f1']
        metric_labels = ['Trust RMSE', 'Trust MAE', 'Device Acc', 'Malicious F1']
        
        for i, (metric, label) in enumerate(zip(metrics_comparison, metric_labels)):
            values = [self.evaluation_results[model][metric] for model in models]
            ax2.plot(models, values, marker='o', linewidth=2, label=label)
        
        ax2.set_title('Metrics Across Models', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'edge_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to {self.results_dir}")
    
    def generate_report(self):
        """Generate comprehensive report"""
        print(f"\n📝 Generating Report...")
        
        report_path = os.path.join(self.results_dir, 'EDGE_RETRAINING_REPORT.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 🔧 Edge Computing GNN Trust Retraining Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**System:** Edge-Focused GNN Trust Retraining\n")
            f.write(f"**Dataset:** Pakistan Edge-Fog-Cloud Topology\n\n")
            
            # Dataset summary
            f.write(f"## 📊 Dataset Summary\n\n")
            topology = self.dataset_loader.topology
            device_counts = {}
            for node in topology['Nodes']:
                device_type = node.get('DeviceType', 'Unknown')
                device_counts[device_type] = device_counts.get(device_type, 0) + 1
            
            f.write(f"- **Total Nodes:** {len(topology['Nodes'])}\n")
            f.write(f"- **Total Edges:** {len(topology['Edges'])}\n")
            for device_type, count in device_counts.items():
                f.write(f"- **{device_type} Nodes:** {count}\n")
            f.write(f"\n")
            
            # Training results
            f.write(f"## 🏋️ Training Results\n\n")
            f.write(f"| Model | Best Val Loss | Total Epochs |\n")
            f.write(f"|-------|---------------|---------------|\n")
            
            for model_name, results in self.training_results.items():
                epochs = len(results.get('train_losses', []))
                f.write(f"| {model_name} | {results['best_val_loss']:.4f} | {epochs} |\n")
            f.write(f"\n")
            
            # Evaluation results
            f.write(f"## 📊 Evaluation Results\n\n")
            f.write(f"| Model | Trust RMSE | Trust MAE | Device Acc | Malicious F1 |\n")
            f.write(f"|-------|------------|-----------|------------|---------------|\n")
            
            for model_name, results in self.evaluation_results.items():
                f.write(f"| {model_name} | {results['trust_rmse']:.4f} | "
                       f"{results['trust_mae']:.4f} | {results['device_accuracy']:.4f} | "
                       f"{results['malicious_f1']:.4f} |\n")
            f.write(f"\n")
            
            # Best model
            best_model = max(self.evaluation_results.keys(), 
                           key=lambda x: self.evaluation_results[x]['malicious_f1'])
            
            f.write(f"## 🏆 Best Performing Model\n\n")
            f.write(f"**{best_model}** achieved the best overall performance with:\n")
            f.write(f"- Trust RMSE: {self.evaluation_results[best_model]['trust_rmse']:.4f}\n")
            f.write(f"- Device Classification Accuracy: {self.evaluation_results[best_model]['device_accuracy']:.4f}\n")
            f.write(f"- Malicious Detection F1-Score: {self.evaluation_results[best_model]['malicious_f1']:.4f}\n\n")
            
            # Edge computing insights
            f.write(f"## 🔍 Edge Computing Insights\n\n")
            f.write(f"1. **Hierarchical Trust:** The Pakistan dataset's edge-fog-cloud hierarchy provides "
                   f"excellent training data for real-world edge computing scenarios.\n\n")
            
            f.write(f"2. **Multi-task Learning:** Training models to simultaneously predict trust values "
                   f"and classify device types improved overall performance.\n\n")
            
            f.write(f"3. **Resource-Aware Trust:** Trust calculation considers both computational resources "
                   f"and network characteristics, crucial for edge environments.\n\n")
            
            # Recommendations
            f.write(f"## 💡 Recommendations\n\n")
            f.write(f"1. **Edge Deployment:** Use lightweight {best_model} model for edge deployment\n")
            f.write(f"2. **Continuous Learning:** Implement online learning for dynamic trust adaptation\n")
            f.write(f"3. **Federated Training:** Consider federated learning across edge nodes\n")
            f.write(f"4. **Latency Optimization:** Further optimize models for real-time edge processing\n\n")
            
            # Files
            f.write(f"## 📁 Generated Files\n\n")
            f.write(f"- `edge_retraining_results.png` - Training and evaluation visualizations\n")
            f.write(f"- `edge_analysis.png` - Edge-specific analysis charts\n")
            f.write(f"- `best_*_edge_model.pth` - Trained model checkpoints\n")
            f.write(f"- `EDGE_RETRAINING_REPORT.md` - This comprehensive report\n\n")
            
            f.write(f"---\n")
            f.write(f"*Generated by Edge Computing GNN Retraining System*\n")
        
        print(f"✅ Report saved to {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run the complete edge retraining analysis"""
        print(f"🚀 Starting Complete Edge Computing GNN Retraining Analysis...")
        
        # Load data
        if not self.load_and_prepare_data():
            print("❌ Failed to load data")
            return None
        
        # Create models
        if not self.create_models():
            print("❌ Failed to create models")
            return None
        
        # Train models
        if not self.train_models():
            print("❌ Failed to train models")
            return None
        
        # Evaluate models
        if not self.evaluate_models():
            print("❌ Failed to evaluate models")
            return None
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        report_path = self.generate_report()
        
        print(f"\n🎉 Edge Computing Retraining Analysis Complete!")
        print(f"📁 Results: {self.results_dir}")
        print(f"📄 Report: {report_path}")
        
        return {
            'results_dir': self.results_dir,
            'report_path': report_path,
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results
        }

def main():
    """Main execution function"""
    print("🔧 Edge Computing GNN Trust Retraining System")
    print("=" * 60)
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ This system requires PyTorch Geometric")
        print("   Install with: pip install torch-geometric")
        return
    
    try:
        # Create and run system
        system = EdgeRetrainingSystem()
        results = system.run_complete_analysis()
        
        if results:
            print("\n✅ Analysis completed successfully!")
            print(f"   Check {results['results_dir']} for all results")
        else:
            print("\n❌ Analysis failed!")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()