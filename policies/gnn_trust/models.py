"""GNN models for trust calculation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math

class BaseGNNModel(nn.Module):
    """Base class for GNN models."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.2, normalize: bool = True):
        """Initialize the model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node embeddings (trust prediction)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            normalize: Whether to use layer normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        
        # Layers will be defined in subclasses
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Trust prediction head
        self.trust_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Trust scores in [0, 1]
        )
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Trust scores [num_nodes, 1]
        """
        # Normalize input features
        x = self.feature_norm(x)
        
        # GNN layers (implemented in subclasses)
        x = self.gnn_forward(x, edge_index, edge_attr)
        
        # Trust prediction
        trust_scores = self.trust_head(x)
        
        return trust_scores
    
    def gnn_forward(self, x, edge_index, edge_attr=None):
        """GNN forward pass - implemented in subclasses."""
        raise NotImplementedError


class GATModel(BaseGNNModel):
    """Graph Attention Network model with proper multi-head attention."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, 
                 num_layers: int = 3, dropout: float = 0.2, heads: int = 8):
        """Initialize the GAT model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (1 for trust score)
            num_layers: Number of GAT layers
            dropout: Dropout rate
            heads: Number of attention heads
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.heads = heads
        self.head_dim = hidden_dim // heads
        
        # Input transformation
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        for i in range(num_layers):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, heads, dropout))
            if self.normalize:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
    
    def gnn_forward(self, x, edge_index, edge_attr=None):
        """GAT forward pass."""
        # Input transformation
        x = F.relu(self.input_transform(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x, edge_index)
            
            # Apply layer norm if enabled
            if self.normalize and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation and dropout
            x = F.relu(x)
            x = self.dropouts[i](x)
            
            # Residual connection
            if x.shape == residual.shape:
                x = x + residual
        
        return x


class GATLayer(nn.Module):
    """Single GAT layer with proper attention mechanism."""
    
    def __init__(self, in_dim: int, out_dim: int, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.dropout = dropout
        
        # Linear transformations for Q, K, V
        self.linear_q = nn.Linear(in_dim, out_dim)
        self.linear_k = nn.Linear(in_dim, out_dim)
        self.linear_v = nn.Linear(in_dim, out_dim)
        
        # Attention weights
        self.attention = nn.Parameter(torch.randn(heads, 2 * self.head_dim))
        
        # Output transformation
        self.output_proj = nn.Linear(out_dim, out_dim)
        
        # Leaky ReLU for attention
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.xavier_uniform_(self.attention)
    
    def forward(self, x, edge_index):
        """Forward pass for GAT layer."""
        num_nodes = x.size(0)
        
        # Linear transformations
        q = self.linear_q(x).view(num_nodes, self.heads, self.head_dim)
        k = self.linear_k(x).view(num_nodes, self.heads, self.head_dim)
        v = self.linear_v(x).view(num_nodes, self.heads, self.head_dim)
        
        # Compute attention scores
        edge_h = self._compute_attention(q, k, edge_index)
        
        # Apply attention to values
        out = self._apply_attention(v, edge_h, edge_index, num_nodes)
        
        # Reshape and apply output projection
        out = out.view(num_nodes, -1)
        out = self.output_proj(out)
        
        return out
    
    def _compute_attention(self, q, k, edge_index):
        """Compute attention scores for edges."""
        if edge_index.numel() == 0:
            return torch.tensor([])
        
        # Get source and target indices
        row, col = edge_index
        
        # Compute attention input: [q_i || k_j] for each edge (i, j)
        attention_input = torch.cat([q[row], k[col]], dim=-1)  # [num_edges, heads, 2*head_dim]
        
        # Compute attention scores
        e = torch.sum(attention_input * self.attention, dim=-1)  # [num_edges, heads]
        e = self.leaky_relu(e)
        
        return e
    
    def _apply_attention(self, v, edge_h, edge_index, num_nodes):
        """Apply attention weights to node values."""
        if edge_index.numel() == 0:
            return v.mean(dim=0, keepdim=True).expand(num_nodes, -1, -1)
        
        row, col = edge_index
        
        # Softmax attention weights for each target node
        attention_weights = torch.zeros(num_nodes, self.heads, device=v.device)
        attention_weights.index_add_(0, col, edge_h)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Apply attention to aggregate neighbor features
        out = torch.zeros_like(v)
        for i in range(num_nodes):
            # Find neighbors
            neighbor_mask = col == i
            if neighbor_mask.any():
                neighbors = row[neighbor_mask]
                neighbor_values = v[neighbors]  # [num_neighbors, heads, head_dim]
                neighbor_weights = edge_h[neighbor_mask]  # [num_neighbors, heads]
                
                # Weighted aggregation
                weighted_values = neighbor_values * neighbor_weights.unsqueeze(-1)
                aggregated = weighted_values.sum(dim=0)  # [heads, head_dim]
                out[i] = aggregated
            else:
                # Self-loop
                out[i] = v[i]
        
        return out


class GraphSAGEModel(BaseGNNModel):
    """GraphSAGE model with proper sampling and aggregation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, 
                 num_layers: int = 3, dropout: float = 0.2, aggr: str = 'mean'):
        """Initialize the GraphSAGE model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (1 for trust score)
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            aggr: Aggregation function ('mean', 'max', 'lstm', or 'pool')
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.aggr = aggr
        
        # Input transformation layer
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        
        # SAGE layers
        dims = [hidden_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            self.layers.append(SAGELayer(in_dim, out_dim, aggr, dropout))
            if self.normalize:
                self.layer_norms.append(nn.LayerNorm(out_dim))
            self.dropouts.append(nn.Dropout(dropout))
    
    def gnn_forward(self, x, edge_index, edge_attr=None):
        """GraphSAGE forward pass."""
        # Input transformation
        x = F.relu(self.input_transform(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # SAGE layers
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x, edge_index)
            
            # Apply layer norm if enabled
            if self.normalize and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation and dropout
            x = F.relu(x)
            x = self.dropouts[i](x)
            
            # Residual connection
            if x.shape == residual.shape:
                x = x + residual
        
        return x


class SAGELayer(nn.Module):
    """Single GraphSAGE layer."""
    
    def __init__(self, in_dim: int, out_dim: int, aggr: str = 'mean', dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggr
        self.dropout = dropout
        
        # Aggregator-specific components
        if aggr == 'lstm':
            self.lstm = nn.LSTM(in_dim, in_dim, batch_first=True)
        elif aggr == 'pool':
            self.pool_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim)
            )
        
        # Self and neighbor transformations
        self.self_transform = nn.Linear(in_dim, out_dim)
        self.neighbor_transform = nn.Linear(in_dim, out_dim)
        
        # Output transformation
        self.output_transform = nn.Linear(out_dim, out_dim)
        
    def forward(self, x, edge_index):
        """Forward pass for SAGE layer."""
        num_nodes = x.size(0)
        
        # Self transformation
        self_emb = self.self_transform(x)
        
        # Neighbor aggregation
        neighbor_emb = self._aggregate_neighbors(x, edge_index)
        neighbor_emb = self.neighbor_transform(neighbor_emb)
        
        # Combine self and neighbor embeddings
        out = self_emb + neighbor_emb
        out = self.output_transform(out)
        
        # L2 normalization
        out = F.normalize(out, p=2, dim=-1)
        
        return out
    
    def _aggregate_neighbors(self, x, edge_index):
        """Aggregate neighbor features."""
        num_nodes = x.size(0)
        
        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        
        row, col = edge_index
        aggregated = torch.zeros_like(x)
        
        for i in range(num_nodes):
            # Find neighbors
            neighbor_mask = col == i
            if neighbor_mask.any():
                neighbors = row[neighbor_mask]
                neighbor_features = x[neighbors]
                
                # Apply aggregation
                if self.aggr == 'mean':
                    agg_feat = neighbor_features.mean(dim=0)
                elif self.aggr == 'max':
                    agg_feat = neighbor_features.max(dim=0)[0]
                elif self.aggr == 'sum':
                    agg_feat = neighbor_features.sum(dim=0)
                elif self.aggr == 'lstm':
                    # LSTM aggregation
                    if len(neighbor_features) > 1:
                        neighbor_features = neighbor_features.unsqueeze(0)  # Add batch dim
                        _, (h_n, _) = self.lstm(neighbor_features)
                        agg_feat = h_n.squeeze(0).squeeze(0)
                    else:
                        agg_feat = neighbor_features.squeeze(0)
                elif self.aggr == 'pool':
                    # Pooling aggregation
                    pooled = self.pool_mlp(neighbor_features)
                    agg_feat = pooled.max(dim=0)[0]
                else:
                    agg_feat = neighbor_features.mean(dim=0)
                
                aggregated[i] = agg_feat
            else:
                # No neighbors, use self
                aggregated[i] = x[i]
        
        return aggregated


class GCNModel(BaseGNNModel):
    """Graph Convolutional Network model with proper spectral convolution."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, 
                 num_layers: int = 3, dropout: float = 0.2, improved: bool = True):
        """Initialize the GCN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (1 for trust score)
            num_layers: Number of GCN layers
            dropout: Dropout rate
            improved: Whether to use improved GCN formulation
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.improved = improved
        
        # Input transformation
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers
        for i in range(num_layers):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, improved))
            if self.normalize:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
    
    def gnn_forward(self, x, edge_index, edge_attr=None):
        """GCN forward pass."""
        # Input transformation
        x = F.relu(self.input_transform(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GCN layers
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x, edge_index)
            
            # Apply layer norm if enabled
            if self.normalize and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation and dropout
            x = F.relu(x)
            x = self.dropouts[i](x)
            
            # Residual connection
            if x.shape == residual.shape:
                x = x + residual
        
        return x


class GCNLayer(nn.Module):
    """Single GCN layer with proper normalization."""
    
    def __init__(self, in_dim: int, out_dim: int, improved: bool = True, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.improved = improved
        
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x, edge_index):
        """Forward pass for GCN layer."""
        # Linear transformation
        x = self.linear(x)
        
        # Graph convolution
        x = self._gcn_conv(x, edge_index)
        
        return x
    
    def _gcn_conv(self, x, edge_index):
        """Apply GCN convolution with proper normalization."""
        num_nodes = x.size(0)
        
        if edge_index.numel() == 0:
            return x
        
        # Add self-loops
        row, col = edge_index
        self_loops = torch.arange(num_nodes, device=edge_index.device)
        edge_index_with_loops = torch.cat([
            edge_index,
            torch.stack([self_loops, self_loops])
        ], dim=1)
        
        row, col = edge_index_with_loops
        
        # Compute node degrees
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        
        # Compute normalization coefficients
        if self.improved:
            # Improved GCN: D^(-1/2) A D^(-1/2)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            # Standard GCN: D^(-1) A
            deg_inv = deg.pow(-1)
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            norm = deg_inv[col]
        
        # Apply convolution
        out = torch.zeros_like(x)
        for i in range(len(row)):
            out[col[i]] += norm[i] * x[row[i]]
        
        return out


class TrustGraphTransformer(BaseGNNModel):
    """Graph Transformer model specifically designed for trust prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, 
                 num_layers: int = 4, dropout: float = 0.2, heads: int = 8):
        """Initialize the Trust Graph Transformer.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (1 for trust score)
            num_layers: Number of transformer layers
            dropout: Dropout rate
            heads: Number of attention heads
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.heads = heads
        
        # Input transformation
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        for i in range(num_layers):
            self.layers.append(GraphTransformerLayer(hidden_dim, heads, dropout))
            if self.normalize:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Trust-specific output layers
        self.trust_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def gnn_forward(self, x, edge_index, edge_attr=None):
        """Graph Transformer forward pass."""
        # Input transformation
        x = F.relu(self.input_transform(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x, edge_index)
            
            # Apply layer norm if enabled
            if self.normalize and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply dropout
            x = self.dropouts[i](x)
            
            # Residual connection
            x = x + residual
        
        return x
    
    def forward(self, x, edge_index, edge_attr=None):
        """Override forward to use trust-specific classifier."""
        # Normalize input features
        x = self.feature_norm(x)
        
        # Graph transformer layers
        x = self.gnn_forward(x, edge_index, edge_attr)
        
        # Trust prediction with specialized classifier
        trust_scores = self.trust_classifier(x)
        
        return trust_scores


class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with structural encoding."""
    
    def __init__(self, hidden_dim: int, heads: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index):
        """Forward pass for transformer layer."""
        # Self-attention with graph structure
        attn_mask = self._create_attention_mask(x.size(0), edge_index)
        
        # Multi-head attention
        x_attended, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0), 
                                      attn_mask=attn_mask)
        x_attended = x_attended.squeeze(0)
        
        # First residual connection
        x = self.norm1(x + x_attended)
        
        # Feed-forward network
        x_ffn = self.ffn(x)
        
        # Second residual connection
        x = self.norm2(x + x_ffn)
        
        return x
    
    def _create_attention_mask(self, num_nodes, edge_index):
        """Create attention mask based on graph structure."""
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        if edge_index.numel() > 0:
            row, col = edge_index
            adj[row, col] = 1
            adj[col, row] = 1  # Make symmetric
        
        # Add self-loops
        adj.fill_diagonal_(1)
        
        # Convert to attention mask (0 for allowed, -inf for masked)
        mask = torch.where(adj == 0, float('-inf'), 0.0)
        
        return mask