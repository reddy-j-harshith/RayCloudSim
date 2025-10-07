"""Aggregation strategies for GNN-based trust calculation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

class BaseAggregator:
    """Base class for aggregation functions."""
    
    def __init__(self, config: Dict = None):
        """Initialize the aggregator.
        
        Args:
            config: Configuration for the aggregator
        """
        self.config = config or {}
    
    def aggregate(self, 
                 features: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate features.
        
        Args:
            features: Features to aggregate [num_nodes, feature_dim]
            mask: Mask for nodes to include/exclude [num_nodes]
            
        Returns:
            Aggregated features
        """
        raise NotImplementedError("Subclasses must implement this method")


class MeanAggregator(BaseAggregator):
    """Mean aggregation function."""
    
    def aggregate(self, 
                 features: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate features by taking the mean.
        
        Args:
            features: Features to aggregate [num_nodes, feature_dim]
            mask: Mask for nodes to include/exclude [num_nodes]
            
        Returns:
            Mean of features
        """
        if mask is not None:
            # Apply mask
            masked_features = features * mask.unsqueeze(1)
            # Compute mean (sum / count)
            count = mask.sum().clamp(min=1)  # Avoid division by zero
            return masked_features.sum(dim=0) / count
        
        return features.mean(dim=0)


class SumAggregator(BaseAggregator):
    """Sum aggregation function."""
    
    def aggregate(self, 
                 features: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate features by taking the sum.
        
        Args:
            features: Features to aggregate [num_nodes, feature_dim]
            mask: Mask for nodes to include/exclude [num_nodes]
            
        Returns:
            Sum of features
        """
        if mask is not None:
            # Apply mask
            masked_features = features * mask.unsqueeze(1)
            return masked_features.sum(dim=0)
        
        return features.sum(dim=0)


class MaxAggregator(BaseAggregator):
    """Max aggregation function."""
    
    def aggregate(self, 
                 features: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate features by taking the max.
        
        Args:
            features: Features to aggregate [num_nodes, feature_dim]
            mask: Mask for nodes to include/exclude [num_nodes]
            
        Returns:
            Max of features
        """
        if mask is not None:
            # Apply mask (set masked-out values to -inf)
            masked_features = features.clone()
            masked_features[mask == 0] = float('-inf')
            return masked_features.max(dim=0)[0]
        
        return features.max(dim=0)[0]


class AttentionAggregator(BaseAggregator):
    """Attention-based aggregation function."""
    
    def __init__(self, config: Dict = None):
        """Initialize the attention aggregator.
        
        Args:
            config: Configuration for the aggregator
                - feature_dim: Dimension of node features
                - num_heads: Number of attention heads
        """
        super().__init__(config)
        
        feature_dim = self.config.get('feature_dim', 16)
        num_heads = self.config.get('num_heads', 4)
        
        # Ensure feature_dim is divisible by num_heads
        if feature_dim % num_heads != 0:
            num_heads = 1  # Fall back to single head
        
        # Create attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Query vector (learned parameter)
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
    
    def aggregate(self, 
                 features: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate features using attention.
        
        Args:
            features: Features to aggregate [num_nodes, feature_dim]
            mask: Mask for nodes to include/exclude [num_nodes]
            
        Returns:
            Attention-weighted features
        """
        # If no features to aggregate
        if features.shape[0] == 0:
            return torch.zeros(1, features.shape[1])
        
        # Add batch dimension for attention
        features = features.unsqueeze(0)  # [1, num_nodes, feature_dim]
        
        # Create attention mask if needed
        attn_mask = None
        if mask is not None:
            # Invert mask for attention (1=attend, 0=ignore)
            attn_mask = ~mask.bool().unsqueeze(0)  # [1, num_nodes]
        
        # Expand query to match batch size
        query = self.query.expand(1, features.shape[1], -1)
        
        # Apply attention
        attended_features, _ = self.attention(
            query=query,
            key=features,
            value=features,
            key_padding_mask=attn_mask
        )
        
        # Remove batch dimension and return
        return attended_features.squeeze(0).mean(dim=0)


class RobustAggregator(BaseAggregator):
    """Robust aggregation function resistant to malicious inputs."""
    
    def __init__(self, config: Dict = None):
        """Initialize the robust aggregator.
        
        Args:
            config: Configuration for the aggregator
                - outlier_threshold: Z-score threshold for outlier detection
        """
        super().__init__(config)
        
        self.outlier_threshold = self.config.get('outlier_threshold', 2.0)
    
    def aggregate(self, 
                 features: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate features robustly by filtering outliers.
        
        Args:
            features: Features to aggregate [num_nodes, feature_dim]
            mask: Mask for nodes to include/exclude [num_nodes]
            
        Returns:
            Robustly aggregated features
        """
        if features.shape[0] <= 1:
            # No need for robust aggregation with 0 or 1 node
            return features.mean(dim=0) if features.shape[0] > 0 else torch.zeros(features.shape[1])
        
        # Apply mask if provided
        if mask is not None:
            # Apply mask
            masked_features = features * mask.unsqueeze(1)
            # Only consider unmasked nodes
            unmasked = mask > 0
            if unmasked.sum() == 0:
                return torch.zeros(features.shape[1])
            features = masked_features[unmasked]
        
        # Compute mean and std for each feature
        mean = features.mean(dim=0)
        std = features.std(dim=0) + 1e-5  # Add small epsilon to avoid division by zero
        
        # Compute z-scores
        z_scores = torch.abs((features - mean) / std)
        
        # Find outliers (features with high z-scores)
        outliers = z_scores > self.outlier_threshold
        
        # Create a mask for valid (non-outlier) features
        valid_mask = ~outliers
        
        # Count valid values per feature
        valid_counts = valid_mask.sum(dim=0)
        
        # Where all values are outliers, keep all values
        all_outliers = valid_counts == 0
        valid_mask[:, all_outliers] = True
        valid_counts[all_outliers] = features.shape[0]
        
        # Compute masked mean
        masked_sum = (features * valid_mask.float()).sum(dim=0)
        robust_mean = masked_sum / valid_counts.float()
        
        return robust_mean


class OpinionAggregator(BaseAggregator):
    """Opinion-based aggregation using similarity matrix."""
    
    def __init__(self, config: Dict = None):
        """Initialize the opinion aggregator.
        
        Args:
            config: Configuration for the aggregator
                - similarity_matrix: Node similarity matrix
                - trust_weight: Weight for trust opinion (vs. own opinion)
        """
        super().__init__(config)
        
        self.similarity_matrix = self.config.get('similarity_matrix', None)
        self.trust_weight = self.config.get('trust_weight', 0.7)
    
    def aggregate(self, 
                 features: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None,
                 node_indices: Optional[List[int]] = None) -> torch.Tensor:
        """Aggregate features based on node opinions/similarities.
        
        Args:
            features: Features to aggregate [num_nodes, feature_dim]
            mask: Mask for nodes to include/exclude [num_nodes]
            node_indices: Indices of nodes in the similarity matrix
            
        Returns:
            Opinion-weighted features
        """
        if self.similarity_matrix is None or features.shape[0] <= 1 or node_indices is None:
            # Fall back to mean aggregation
            return features.mean(dim=0) if features.shape[0] > 0 else torch.zeros(features.shape[1])
        
        # Apply mask if provided
        if mask is not None:
            # Apply mask
            masked_features = features * mask.unsqueeze(1)
            # Only consider unmasked nodes
            unmasked = mask > 0
            if unmasked.sum() == 0:
                return torch.zeros(features.shape[1])
            features = masked_features[unmasked]
            # Update node indices for unmasked nodes
            if node_indices is not None:
                node_indices = [idx for i, idx in enumerate(node_indices) if unmasked[i]]
        
        # Get similarity submatrix for the nodes
        sim_submatrix = self.similarity_matrix[node_indices][:, node_indices]
        
        # Convert to tensor if necessary
        if not isinstance(sim_submatrix, torch.Tensor):
            sim_submatrix = torch.tensor(sim_submatrix, dtype=torch.float)
        
        # Normalize similarities
        weights = F.softmax(sim_submatrix, dim=1)
        
        # Weight features by opinions
        weighted_features = torch.matmul(weights, features)
        
        # Combine own features with opinion-weighted features
        result = (1 - self.trust_weight) * features + self.trust_weight * weighted_features
        
        # Return mean across nodes
        return result.mean(dim=0)