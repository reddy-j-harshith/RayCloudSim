"""Training system for GNN trust regression models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict
import time

from .models import GATModel, GraphSAGEModel, GCNModel, TrustGraphTransformer
from policies.gnn_trust.feature_engineering_simple import extract_node_features

class TrustRegressionDataset(Dataset):
    """Dataset for trust value regression tasks."""
    
    def __init__(self, graphs: List[nx.Graph], trust_values: List[Dict[str, float]], malicious_labels: List[Dict[str, int]] = None):
        """Initialize the trust regression dataset.
        
        Args:
            graphs: List of NetworkX graphs
            trust_values: List of continuous trust value dictionaries for each graph [0,1]
            malicious_labels: List of malicious label dictionaries (for evaluation only)
        """
        self.graphs = graphs
        self.trust_values = trust_values
        self.malicious_labels = malicious_labels if malicious_labels else []
        
        # Pre-process all graphs
        self.processed_data = []
        for i, (graph, trust_vals) in enumerate(zip(graphs, trust_values)):
            mal_labels = self.malicious_labels[i] if i < len(self.malicious_labels) else {}
            processed = self._process_graph(graph, trust_vals, mal_labels)
            if processed is not None:
                self.processed_data.append(processed)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
    
    def _process_graph(self, graph: nx.Graph, trust_values: Dict[str, float], malicious_labels: Dict[str, int] = None):
        """Process a single graph into model inputs for regression."""
        try:
            node_ids = list(graph.nodes())
            if len(node_ids) == 0:
                return None
            
            # Extract node features using the simplified function
            combined_features = extract_node_features(graph)
            
            # Create edge indices
            edge_list = list(graph.edges())
            if not edge_list:
                # Create self-loops if no edges
                edge_list = [(node, node) for node in node_ids]
            
            edge_indices = torch.tensor([[node_ids.index(src), node_ids.index(dst)] 
                                        for src, dst in edge_list], dtype=torch.long).t()
            
            # Create trust values tensor (continuous regression targets)
            trust_tensor = torch.tensor([trust_values.get(node, 0.5) for node in node_ids], 
                                       dtype=torch.float)
            
            # Create malicious labels tensor (for evaluation)
            malicious_tensor = None
            if malicious_labels:
                malicious_tensor = torch.tensor([malicious_labels.get(node, 0) for node in node_ids], 
                                               dtype=torch.long)
            
            return {
                'node_features': combined_features,
                'edge_index': edge_indices,
                'trust_values': trust_tensor,  # Continuous values [0,1]
                'malicious_labels': malicious_tensor,  # Binary labels for evaluation
                'node_ids': node_ids
            }
        except Exception as e:
            print(f"Error processing graph: {e}")
            return None


class TrustRegressionTrainer:
    """Trainer for GNN trust regression models."""
    
    def __init__(self, model_type: str = 'gat', config: Dict = None):
        """Initialize the regression trainer.
        
        Args:
            model_type: Type of GNN model ('gat', 'graphsage', 'gcn', 'transformer')
            config: Training configuration
        """
        self.model_type = model_type
        
        # Default configuration for regression
        default_config = {
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 1,  # Usually 1 graph at a time
            'num_epochs': 200,
            'early_stopping_patience': 20,
            'val_split': 0.2,
            'test_split': 0.1,
            'use_scheduler': True,
            'scheduler_patience': 10,
            'scheduler_factor': 0.5,
            'min_lr': 1e-6,
            'trust_threshold': 0.5,  # Threshold for malicious detection
            'gradient_clip': 1.0,
            'save_dir': 'checkpoints/gnn_trust_regression',
            'log_interval': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize components
        self.device = torch.device(self.config['device'])
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        
        # Create save directory
        os.makedirs(self.config['save_dir'], exist_ok=True)
    
    def create_model(self, input_dim: int):
        """Create the GNN regression model.
        
        Args:
            input_dim: Input feature dimension
        """
        # Modify models to output single continuous value with sigmoid activation
        if self.model_type == 'gat':
            self.model = GATModel(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=1,  # Single trust value output
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                heads=8
            ).to(self.device)
        elif self.model_type == 'graphsage':
            self.model = GraphSAGEModel(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=1,
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                aggr='mean'
            ).to(self.device)
        elif self.model_type == 'gcn':
            self.model = GCNModel(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=1,
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
        elif self.model_type == 'transformer':
            self.model = TrustGraphTransformer(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=1,
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                heads=8
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Add sigmoid activation to ensure output is in [0,1]
        self.model = nn.Sequential(
            self.model,
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Initialize scheduler
        if self.config['use_scheduler']:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config['scheduler_patience'],
                factor=self.config['scheduler_factor'],
                min_lr=self.config['min_lr']
            )
        
        print(f"Created {self.model_type.upper()} regression model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def prepare_datasets(self, graphs: List[nx.Graph], trust_values: List[Dict[str, float]], malicious_labels: List[Dict[str, int]] = None):
        """Prepare training, validation, and test datasets.
        
        Args:
            graphs: List of NetworkX graphs
            trust_values: List of trust value dictionaries
            malicious_labels: List of malicious label dictionaries
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create dataset
        dataset = TrustRegressionDataset(graphs, trust_values, malicious_labels)
        
        if len(dataset) == 0:
            raise ValueError("No valid samples in dataset")
        
        # Split dataset
        total_split = self.config['val_split'] + self.config['test_split']
        if total_split > 0:
            train_indices, temp_indices = train_test_split(
                range(len(dataset)), 
                test_size=total_split,
                random_state=42
            )
            
            if temp_indices and self.config['val_split'] > 0:
                val_size = self.config['val_split'] / total_split
                val_indices, test_indices = train_test_split(temp_indices, test_size=1-val_size, random_state=42)
            else:
                val_indices, test_indices = temp_indices, []
        else:
            # No validation or test split
            train_indices = list(range(len(dataset)))
            val_indices, test_indices = [], []
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices) if val_indices else None
        test_dataset = torch.utils.data.Subset(dataset, test_indices) if test_indices else None
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=self._collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=self._collate_fn) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=self._collate_fn) if test_dataset else None
        
        print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
        
        return train_loader, val_loader, test_loader
    
    def _collate_fn(self, batch):
        """Custom collate function for trust regression data."""
        if len(batch) == 1:
            return batch[0]
        
        # For batch size > 1, would need to handle batching
        # For now, we assume batch_size=1
        return batch[0]
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        for batch_idx, data in enumerate(train_loader):
            # Move data to device
            node_features = data['node_features'].to(self.device)
            edge_index = data['edge_index'].to(self.device)
            trust_values = data['trust_values'].to(self.device)  # Continuous targets
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(node_features, edge_index).squeeze()
            
            # Regression loss (MSE)
            loss = nn.MSELoss()(outputs, trust_values)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_samples += 1
        
        return total_loss / num_samples if num_samples > 0 else 0.0
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on regression and detection metrics."""
        if not data_loader:
            return {}
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_malicious_preds = []
        all_malicious_targets = []
        
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for data in data_loader:
                # Move data to device
                node_features = data['node_features'].to(self.device)
                edge_index = data['edge_index'].to(self.device)
                trust_values = data['trust_values'].to(self.device)
                
                # Forward pass
                outputs = self.model(node_features, edge_index).squeeze()
                
                # Regression loss
                loss = nn.MSELoss()(outputs, trust_values)
                total_loss += loss.item()
                num_samples += 1
                
                # Collect predictions and targets
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(trust_values.cpu().numpy())
                
                # Malicious detection using threshold
                if data['malicious_labels'] is not None:
                    predicted_malicious = (outputs < self.config['trust_threshold']).long()
                    all_malicious_preds.extend(predicted_malicious.cpu().numpy())
                    all_malicious_targets.extend(data['malicious_labels'].cpu().numpy())
        
        # Regression metrics
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'loss': total_loss / num_samples if num_samples > 0 else 0.0,
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
        
        # Malicious detection metrics (if available)
        if all_malicious_targets:
            detection_acc = accuracy_score(all_malicious_targets, all_malicious_preds)
            detection_prec = precision_score(all_malicious_targets, all_malicious_preds, zero_division=0)
            detection_recall = recall_score(all_malicious_targets, all_malicious_preds, zero_division=0)
            detection_f1 = f1_score(all_malicious_targets, all_malicious_preds, zero_division=0)
            
            metrics.update({
                'detection_accuracy': detection_acc,
                'detection_precision': detection_prec,
                'detection_recall': detection_recall,
                'detection_f1': detection_f1
            })
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Complete training loop."""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(self.config['num_epochs']):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_metrics = {}
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics.get('loss', float('inf'))
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_model('best_model.pth')
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % self.config['log_interval'] == 0:
                log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.6f}"
                if val_metrics:
                    log_msg += f", Val Loss = {val_metrics.get('loss', 0):.6f}"
                    log_msg += f", Val RMSE = {val_metrics.get('rmse', 0):.6f}"
                    if 'detection_accuracy' in val_metrics:
                        log_msg += f", Detection Acc = {val_metrics['detection_accuracy']:.4f}"
                print(log_msg)
            
            # Store training history
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
        
        print("Training completed!")
    
    def save_model(self, filename: str):
        """Save model state."""
        filepath = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': dict(self.training_history)
        }, filepath)
    
    def load_model(self, filename: str):
        """Load model state."""
        filepath = os.path.join(self.config['save_dir'], filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = defaultdict(list, checkpoint.get('training_history', {}))
    
    def predict_trust_values(self, graphs: List[nx.Graph]) -> List[Dict[str, float]]:
        """Predict trust values for given graphs."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for graph in graphs:
                # Process graph
                node_ids = list(graph.nodes())
                if not node_ids:
                    predictions.append({})
                    continue
                
                try:
                    # Extract features
                    node_features = extract_node_features(graph)
                    
                    # Create edge indices
                    edge_list = list(graph.edges())
                    if not edge_list:
                        edge_list = [(node, node) for node in node_ids]
                    
                    edge_indices = torch.tensor([[node_ids.index(src), node_ids.index(dst)] 
                                                for src, dst in edge_list], dtype=torch.long).t()
                    
                    # Move to device
                    node_features = node_features.to(self.device)
                    edge_indices = edge_indices.to(self.device)
                    
                    # Predict
                    trust_outputs = self.model(node_features, edge_indices).squeeze()
                    
                    # Convert to dictionary
                    trust_dict = {}
                    if trust_outputs.dim() == 0:  # Single node
                        trust_dict[node_ids[0]] = trust_outputs.item()
                    else:
                        for i, node_id in enumerate(node_ids):
                            trust_dict[node_id] = trust_outputs[i].item()
                    
                    predictions.append(trust_dict)
                    
                except Exception as e:
                    print(f"Error predicting for graph: {e}")
                    predictions.append({node_id: 0.5 for node_id in node_ids})
        
        return predictions
    
    def detect_malicious_nodes(self, graphs: List[nx.Graph], method: str = 'threshold', threshold: float = None) -> List[Dict[str, bool]]:
        """Detect malicious nodes using different methods."""
        if threshold is None:
            threshold = self.config['trust_threshold']
        
        # Get trust predictions
        trust_predictions = self.predict_trust_values(graphs)
        malicious_predictions = []
        
        for trust_dict in trust_predictions:
            if not trust_dict:
                malicious_predictions.append({})
                continue
            
            trust_values = list(trust_dict.values())
            
            if method == 'threshold':
                # Simple threshold-based detection
                malicious_dict = {node: trust < threshold for node, trust in trust_dict.items()}
                
            elif method == 'zscore':
                # Z-score based detection
                mean_trust = np.mean(trust_values)
                std_trust = np.std(trust_values)
                malicious_dict = {}
                for node, trust in trust_dict.items():
                    if std_trust > 0:
                        z_score = abs((trust - mean_trust) / std_trust)
                        malicious_dict[node] = (trust < mean_trust) and (z_score > threshold)
                    else:
                        malicious_dict[node] = trust < 0.5
                        
            elif method == 'percentile':
                # Bottom percentile as malicious
                percentile_threshold = np.percentile(trust_values, threshold * 100)
                malicious_dict = {node: trust <= percentile_threshold for node, trust in trust_dict.items()}
                
            else:
                raise ValueError(f"Unknown detection method: {method}")
            
            malicious_predictions.append(malicious_dict)
        
        return malicious_predictions