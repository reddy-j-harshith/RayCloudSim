"""Training system for GNN trust models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict
import time

from .models import GATModel, GraphSAGEModel, GCNModel, TrustGraphTransformer
from policies.gnn_trust.feature_engineering_simple import extract_node_features

class TrustDataset(Dataset):
    """Dataset for trust prediction tasks."""
    
    def __init__(self, graphs: List[nx.Graph], trust_labels: List[Dict[str, float]]):
        """Initialize the trust dataset.
        
        Args:
            graphs: List of NetworkX graphs
            trust_labels: List of trust label dictionaries for each graph
        """
        self.graphs = graphs
        self.trust_labels = trust_labels
        
        # Pre-process all graphs
        self.processed_data = []
        for i, (graph, labels) in enumerate(zip(graphs, trust_labels)):
            processed = self._process_graph(graph, labels)
            if processed is not None:
                self.processed_data.append(processed)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
    
    def _process_graph(self, graph: nx.Graph, trust_labels: Dict[str, float]):
        """Process a single graph into model inputs."""
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
            
            # Extract edge features (optional)
            edge_features = None
            
            # Create trust labels tensor
            trust_tensor = torch.tensor([trust_labels.get(node, 0.5) for node in node_ids], 
                                       dtype=torch.float)
            
            return {
                'node_features': combined_features,
                'edge_index': edge_indices,
                'edge_features': edge_features,
                'trust_labels': trust_tensor,
                'node_ids': node_ids
            }
        except Exception as e:
            print(f"Error processing graph: {e}")
            return None


class TrustTrainer:
    """Trainer for GNN trust models."""
    
    def __init__(self, model_type: str = 'gat', config: Dict = None):
        """Initialize the trainer.
        
        Args:
            model_type: Type of GNN model ('gat', 'graphsage', 'gcn', 'transformer')
            config: Training configuration
        """
        self.model_type = model_type
        
        # Default configuration
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
            'trust_threshold': 0.5,
            'pos_weight': 2.0,  # Weight for positive class in BCE loss
            'gradient_clip': 1.0,
            'save_dir': 'checkpoints/gnn_trust',
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
        """Create the GNN model.
        
        Args:
            input_dim: Input feature dimension
        """
        if self.model_type == 'gat':
            self.model = GATModel(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=1,
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
        
        print(f"Created {self.model_type.upper()} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def prepare_datasets(self, graphs: List[nx.Graph], trust_labels: List[Dict[str, float]]):
        """Prepare training, validation, and test datasets.
        
        Args:
            graphs: List of NetworkX graphs
            trust_labels: List of trust label dictionaries
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create dataset
        dataset = TrustDataset(graphs, trust_labels)
        
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
        """Collate function for batching graphs."""
        # For now, just return the first item since we're using batch_size=1
        return batch[0]
    
    def train(self, train_loader, val_loader=None):
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        # Define loss function with class weighting
        pos_weight = torch.tensor([self.config['pos_weight']]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, criterion)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_metrics = 0.0, {}
            if val_loader:
                val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
                self.val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_loss if val_loader else train_loss)
            
            # Log progress
            if epoch % self.config['log_interval'] == 0:
                print(f"Epoch {epoch:3d}/{self.config['num_epochs']} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics.get('accuracy', 0):.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics.get('accuracy', 0):.4f}")
            
            # Save training history
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            for metric, value in train_metrics.items():
                self.training_history[f'train_{metric}'].append(value)
            for metric, value in val_metrics.items():
                self.training_history[f'val_{metric}'].append(value)
            
            # Early stopping
            if val_loader:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_model('best_model.pt')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config['early_stopping_patience']:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        print("Training completed!")
        
        # Load best model if available
        if val_loader and os.path.exists(os.path.join(self.config['save_dir'], 'best_model.pt')):
            self.load_model('best_model.pt')
            print("Loaded best model from validation")
    
    def _train_epoch(self, train_loader, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            node_features = batch['node_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_features = batch['edge_features'].to(self.device) if batch['edge_features'] is not None else None
            trust_labels = batch['trust_labels'].to(self.device)
            
            # Make predictions
            predictions = self.model(node_features, edge_index, edge_features)
            predictions = predictions.squeeze(-1)
            
            # Calculate loss
            loss = criterion(predictions, trust_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Convert predictions to probabilities and binary predictions
            probs = torch.sigmoid(predictions)
            binary_preds = (probs > self.config['trust_threshold']).float()
            
            all_predictions.extend(binary_preds.cpu().numpy())
            all_labels.extend(trust_labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_labels, all_predictions)
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Forward pass
                node_features = batch['node_features'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_features = batch['edge_features'].to(self.device) if batch['edge_features'] is not None else None
                trust_labels = batch['trust_labels'].to(self.device)
                
                # Make predictions
                predictions = self.model(node_features, edge_index, edge_features)
                predictions = predictions.squeeze(-1)
                
                # Calculate loss
                loss = criterion(predictions, trust_labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                
                # Convert predictions to probabilities and binary predictions
                probs = torch.sigmoid(predictions)
                binary_preds = (probs > self.config['trust_threshold']).float()
                
                all_predictions.extend(binary_preds.cpu().numpy())
                all_labels.extend(trust_labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_labels, all_predictions)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, labels, predictions):
        """Calculate evaluation metrics."""
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        # Convert to binary labels for classification metrics
        binary_labels = (labels > self.config['trust_threshold']).astype(int)
        binary_preds = predictions.astype(int)
        
        metrics = {}
        try:
            metrics['accuracy'] = accuracy_score(binary_labels, binary_preds)
            metrics['precision'] = precision_score(binary_labels, binary_preds, zero_division=0)
            metrics['recall'] = recall_score(binary_labels, binary_preds, zero_division=0)
            metrics['f1'] = f1_score(binary_labels, binary_preds, zero_division=0)
            
            # ROC-AUC if we have both classes
            if len(np.unique(binary_labels)) > 1:
                metrics['auc'] = roc_auc_score(binary_labels, predictions)
            else:
                metrics['auc'] = 0.5
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0.5}
        
        return metrics
    
    def test(self, test_loader):
        """Test the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Forward pass
                node_features = batch['node_features'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_features = batch['edge_features'].to(self.device) if batch['edge_features'] is not None else None
                trust_labels = batch['trust_labels'].to(self.device)
                
                # Make predictions
                predictions = self.model(node_features, edge_index, edge_features)
                predictions = predictions.squeeze(-1)
                
                # Convert predictions to probabilities and binary predictions
                probs = torch.sigmoid(predictions)
                binary_preds = (probs > self.config['trust_threshold']).float()
                
                all_predictions.extend(binary_preds.cpu().numpy())
                all_labels.extend(trust_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_metrics = self._calculate_metrics(all_labels, all_predictions)
        
        print("Test Results:")
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return test_metrics
    
    def predict(self, graph: nx.Graph) -> Dict[str, float]:
        """Make predictions on a single graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping node IDs to trust scores
        """
        self.model.eval()
        
        # Process graph
        dataset = TrustDataset([graph], [{}])
        if len(dataset) == 0:
            return {}
        
        batch = dataset[0]
        
        with torch.no_grad():
            node_features = batch['node_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_features = batch['edge_features'].to(self.device) if batch['edge_features'] is not None else None
            
            # Make predictions
            predictions = self.model(node_features, edge_index, edge_features)
            probs = torch.sigmoid(predictions.squeeze(-1))
            
            # Create result dictionary
            trust_scores = {}
            for i, node_id in enumerate(batch['node_ids']):
                trust_scores[node_id] = float(probs[i].cpu().numpy())
        
        return trust_scores
    
    def save_model(self, filename: str):
        """Save the model and training state."""
        if self.model is None:
            return
        
        save_path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'training_history': dict(self.training_history),
            'best_val_loss': self.best_val_loss
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load the model and training state."""
        load_path = os.path.join(self.config['save_dir'], filename)
        
        if not os.path.exists(load_path):
            print(f"Model file {load_path} not found")
            return False
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Update config
        self.config.update(checkpoint['config'])
        
        # Create model if not exists
        if self.model is None:
            # Need to determine input dimension from saved model
            sample_weight = list(checkpoint['model_state_dict'].values())[0]
            if 'input_transform.weight' in checkpoint['model_state_dict']:
                input_dim = checkpoint['model_state_dict']['input_transform.weight'].shape[1]
            elif 'feature_norm.weight' in checkpoint['model_state_dict']:
                input_dim = checkpoint['model_state_dict']['feature_norm.weight'].shape[0]
            else:
                input_dim = 64  # fallback
            
            self.create_model(input_dim)
        
        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        if 'training_history' in checkpoint:
            self.training_history = defaultdict(list, checkpoint['training_history'])
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Model loaded from {load_path}")
        return True
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        if not self.training_history['epoch']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss'] and any(x > 0 for x in self.training_history['val_loss']):
            axes[0, 0].plot(self.training_history['epoch'], self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        if 'train_accuracy' in self.training_history:
            axes[0, 1].plot(self.training_history['epoch'], self.training_history['train_accuracy'], label='Train Acc')
            if 'val_accuracy' in self.training_history:
                axes[0, 1].plot(self.training_history['epoch'], self.training_history['val_accuracy'], label='Val Acc')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # F1 Score plot
        if 'train_f1' in self.training_history:
            axes[1, 0].plot(self.training_history['epoch'], self.training_history['train_f1'], label='Train F1')
            if 'val_f1' in self.training_history:
                axes[1, 0].plot(self.training_history['epoch'], self.training_history['val_f1'], label='Val F1')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('Training and Validation F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # AUC plot
        if 'train_auc' in self.training_history:
            axes[1, 1].plot(self.training_history['epoch'], self.training_history['train_auc'], label='Train AUC')
            if 'val_auc' in self.training_history:
                axes[1, 1].plot(self.training_history['epoch'], self.training_history['val_auc'], label='Val AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].set_title('Training and Validation AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.config['save_dir'], 'training_history.png'), dpi=300, bbox_inches='tight')
        
        plt.show()