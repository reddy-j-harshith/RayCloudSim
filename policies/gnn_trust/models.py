"""GNN models for trust calculation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class BaseGNNModel(nn.Module):
    """Base class for GNN models."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        """Initialize the model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node embeddings
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Layers will be defined in subclasses
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Implemented in subclasses
        pass


class GATModel(BaseGNNModel):
    """Graph Attention Network model (simplified implementation)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1, heads: int = 4):
        """Initialize the GAT model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node embeddings
            num_layers: Number of GAT layers
            dropout: Dropout rate
            heads: Number of attention heads
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.heads = heads
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer: hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Process through linear layers with attention-like aggregation
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.batch_norms)):
            # Apply linear transformation
            x = layer(x)
            
            # Apply attention-like aggregation using neighbors
            if edge_index.numel() > 0:
                x = self._apply_attention_aggregation(x, edge_index)
            
            # Apply batch norm and activation
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.layers[-1](x)
        return x
    
    def _apply_attention_aggregation(self, x, edge_index):
        """Apply attention-like aggregation based on graph structure."""
        # Simple aggregation based on graph connectivity
        num_nodes = x.size(0)
        aggregated = x.clone()
        
        # For each node, aggregate features from neighbors
        for i in range(num_nodes):
            # Find neighbors
            neighbor_mask = edge_index[1] == i
            if neighbor_mask.any():
                neighbors = edge_index[0][neighbor_mask]
                neighbor_features = x[neighbors]
                
                # Simple attention-like weighting (could be improved)
                weights = F.softmax(torch.randn(len(neighbors)), dim=0)
                aggregated[i] = (aggregated[i] + (neighbor_features * weights.unsqueeze(1)).sum(dim=0)) / 2
        
        return aggregated


class GraphSAGEModel(BaseGNNModel):
    """GraphSAGE model (simplified implementation)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1, aggr: str = 'mean'):
        """Initialize the GraphSAGE model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node embeddings
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            aggr: Aggregation function ('mean', 'max', or 'sum')
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.aggr = aggr
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(nn.Linear(input_dim * 2, hidden_dim))  # self + neighbor features
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer: hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim * 2, output_dim))
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.batch_norms)):
            # Apply SAGE-like aggregation
            x = self._sage_aggregation(x, edge_index, layer)
            # Apply batch norm and activation
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self._sage_aggregation(x, edge_index, self.layers[-1])
        return x
    
    def _sage_aggregation(self, x, edge_index, layer):
        """Apply SAGE-like aggregation."""
        num_nodes = x.size(0)
        output = []
        
        for i in range(num_nodes):
            # Get self features
            self_feat = x[i]
            
            # Find neighbors
            neighbor_mask = edge_index[1] == i
            if neighbor_mask.any():
                neighbors = edge_index[0][neighbor_mask]
                neighbor_features = x[neighbors]
                
                # Aggregate neighbor features
                if self.aggr == 'mean':
                    neighbor_agg = neighbor_features.mean(dim=0)
                elif self.aggr == 'max':
                    neighbor_agg = neighbor_features.max(dim=0)[0]
                elif self.aggr == 'sum':
                    neighbor_agg = neighbor_features.sum(dim=0)
                else:
                    neighbor_agg = neighbor_features.mean(dim=0)
            else:
                neighbor_agg = torch.zeros_like(self_feat)
            
            # Combine self and neighbor features
            combined = torch.cat([self_feat, neighbor_agg])
            output.append(layer(combined))
        
        return torch.stack(output)


class GCNModel(BaseGNNModel):
    """Graph Convolutional Network model (simplified implementation)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        """Initialize the GCN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node embeddings
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer: hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.batch_norms)):
            # Apply linear transformation
            x = layer(x)
            
            # Apply GCN-like aggregation
            if edge_index.numel() > 0:
                x = self._gcn_aggregation(x, edge_index)
            
            # Apply batch norm and activation
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.layers[-1](x)
        if edge_index.numel() > 0:
            x = self._gcn_aggregation(x, edge_index)
        return x
    
    def _gcn_aggregation(self, x, edge_index):
        """Apply GCN-like aggregation."""
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(x)
        
        # Count degrees for normalization
        degrees = torch.zeros(num_nodes)
        for i in range(num_nodes):
            neighbor_mask = edge_index[1] == i
            degrees[i] = neighbor_mask.sum().float() + 1  # +1 for self-loop
        
        # Aggregate features
        for i in range(num_nodes):
            # Self-loop
            aggregated[i] += x[i] / torch.sqrt(degrees[i])
            
            # Neighbors
            neighbor_mask = edge_index[1] == i
            if neighbor_mask.any():
                neighbors = edge_index[0][neighbor_mask]
                for neighbor in neighbors:
                    aggregated[i] += x[neighbor] / torch.sqrt(degrees[i] * degrees[neighbor])
        
        return aggregated