"""Message passing mechanisms for GNN-based trust calculation."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union

class MessagePassing:
    """Message passing for trust propagation in the network."""
    
    def __init__(self, config: Dict = None):
        """Initialize the message passing mechanism.
        
        Args:
            config: Configuration for message passing
                - max_hops: Maximum number of hops for message passing
                - use_edge_weights: Whether to use edge weights
                - temporal_decay: Decay factor for temporal messages
                - aggr_mode: Aggregation mode ('mean', 'sum', 'max', 'attention')
        """
        self.config = config or {
            'max_hops': 2,
            'use_edge_weights': True,
            'temporal_decay': 0.8,
            'aggr_mode': 'attention'
        }
    
    def propagate_trust(self, 
                         node_features: torch.Tensor, 
                         edge_index: torch.Tensor, 
                         edge_attr: Optional[torch.Tensor] = None,
                         node_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Propagate trust information through the network.
        
        Args:
            node_features: Node feature tensor [num_nodes, feature_dim]
            edge_index: Edge index tensor [2, num_edges]
            edge_attr: Edge attribute tensor [num_edges, edge_dim]
            node_masks: Masks for nodes (e.g., for new nodes) [num_nodes]
            
        Returns:
            Updated node features
        """
        # Create a custom message passing layer
        mp_layer = TrustMessagePassing(
            node_features.shape[1], 
            aggr=self.config['aggr_mode'],
            use_edge_weights=self.config['use_edge_weights']
        )
        
        # Initialize with original features
        h = node_features
        
        # Propagate for multiple hops
        for hop in range(self.config['max_hops']):
            # Apply temporal decay to previous hop's features
            if hop > 0:
                h = h * self.config['temporal_decay']
            
            # Propagate features
            h = mp_layer(h, edge_index, edge_attr)
            
            # Apply node masks if provided
            if node_masks is not None:
                h = h * node_masks.unsqueeze(1)
        
        return h
    
    def initialize_new_node(self, 
                           node_id: str, 
                           neighbor_features: torch.Tensor, 
                           edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Initialize features for a new node based on its neighbors.
        
        Args:
            node_id: ID of the new node
            neighbor_features: Features of neighboring nodes [num_neighbors, feature_dim]
            edge_attr: Edge attributes for connections to neighbors [num_neighbors, edge_dim]
            
        Returns:
            Initialized node features
        """
        # If no neighbors, return zeros
        if len(neighbor_features) == 0:
            return torch.zeros(1, neighbor_features.shape[1])
        
        # Weight neighbor features by edge weights if available
        if edge_attr is not None and self.config['use_edge_weights']:
            # Use first dimension of edge attributes as weights
            weights = torch.softmax(edge_attr[:, 0], dim=0)
            weighted_features = neighbor_features * weights.unsqueeze(1)
            return weighted_features.mean(dim=0, keepdim=True)
        
        # Otherwise, just take the mean
        return neighbor_features.mean(dim=0, keepdim=True)


class TrustMessagePassing(nn.Module):
    """Custom message passing layer for trust propagation."""
    
    def __init__(self, feature_dim: int, aggr: str = 'mean', use_edge_weights: bool = True):
        """Initialize the trust message passing layer.
        
        Args:
            feature_dim: Dimension of node features
            aggr: Aggregation function ('mean', 'sum', 'max', 'attention')
            use_edge_weights: Whether to use edge weights
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.aggr_mode = aggr
        self.use_edge_weights = use_edge_weights
        
        # For attention-based aggregation
        if aggr == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 1)
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge index [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Updated node features
        """
        # Process edge attributes if provided
        edge_weight = None
        if edge_attr is not None and self.use_edge_weights:
            # Use first dimension of edge attributes as weights
            edge_weight = edge_attr[:, 0]
        
        # Manual message passing
        return self._propagate_messages(x, edge_index, edge_weight)
    
    def _propagate_messages(self, x: torch.Tensor, edge_index: torch.Tensor, 
                           edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Manually propagate messages.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge index [2, num_edges]
            edge_weight: Edge weights [num_edges]
            
        Returns:
            Updated node features
        """
        num_nodes = x.size(0)
        updated_features = x.clone()
        
        # Get source and target nodes
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Collect messages for each node
        for i in range(num_nodes):
            # Find incoming edges for node i
            incoming_mask = target_nodes == i
            if not incoming_mask.any():
                continue
            
            # Get source nodes and features
            source_indices = source_nodes[incoming_mask]
            source_features = x[source_indices]  # [num_incoming, feature_dim]
            target_feature = x[i:i+1].expand_as(source_features)  # [num_incoming, feature_dim]
            
            # Apply message function
            messages = self._message(source_features, target_feature, 
                                   edge_weight[incoming_mask] if edge_weight is not None else None)
            
            # Aggregate messages
            if self.aggr_mode == 'mean':
                aggregated = messages.mean(dim=0)
            elif self.aggr_mode == 'sum' or self.aggr_mode == 'attention':
                aggregated = messages.sum(dim=0)
            elif self.aggr_mode == 'max':
                aggregated = messages.max(dim=0)[0]
            else:
                aggregated = messages.mean(dim=0)
            
            # Update node features
            updated_features[i] = x[i] + aggregated
        
        return updated_features
    
    def _message(self, x_j: torch.Tensor, x_i: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Define message function.
        
        Args:
            x_j: Source node features [num_messages, feature_dim]
            x_i: Target node features [num_messages, feature_dim]
            edge_weight: Edge weights [num_messages]
            
        Returns:
            Messages to pass [num_messages, feature_dim]
        """
        # Apply attention if specified
        if self.aggr_mode == 'attention':
            # Compute attention weights
            attention_input = torch.cat([x_i, x_j], dim=1)
            attention_weight = torch.sigmoid(self.attention(attention_input))
            
            # Apply edge weights if available
            if edge_weight is not None:
                attention_weight = attention_weight * edge_weight.view(-1, 1)
            
            return x_j * attention_weight
        
        # Otherwise, apply edge weights directly
        if edge_weight is not None:
            return x_j * edge_weight.view(-1, 1)
        
        return x_j