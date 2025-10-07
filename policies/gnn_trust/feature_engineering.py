"""Feature engineering for GNN-based trust calculation."""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx

class FeatureExtractor:
    """Extract features from nodes for GNN-based trust calculation."""
    
    def __init__(self, feature_config: Dict = None):
        """Initialize the feature extractor.
        
        Args:
            feature_config: Configuration for feature extraction
                - temporal_window: Window size for temporal features
                - node_features: List of node features to extract
                - edge_features: List of edge features to extract
                - use_spectral: Whether to use spectral embedding
                - embedding_dim: Dimension of node embeddings
        """
        self.feature_config = feature_config or {
            'temporal_window': 10,
            'node_features': ['cpu_freq', 'buffer_size', 'energy_level', 'task_count'],
            'edge_features': ['latency', 'bandwidth', 'trust_score'],
            'use_spectral': True,
            'embedding_dim': 16
        }
        
        self.history_buffer = {}  # Store temporal history for each node
        self.last_embeddings = {}  # Cache for the last computed embeddings
        
    def extract_node_features(self, graph: nx.Graph, nodes: List) -> torch.Tensor:
        """Extract node features from the graph.
        
        Args:
            graph: NetworkX graph
            nodes: List of node IDs
            
        Returns:
            Tensor of node features
        """
        features = []
        
        for node in nodes:
            # Get node data
            node_data = graph.nodes[node].get('data', {})
            if not isinstance(node_data, dict):
                node_data = {'node_id': node}
                
            # Extract static features
            node_feats = []
            for feat in self.feature_config['node_features']:
                if hasattr(node_data, feat):
                    node_feats.append(getattr(node_data, feat))
                elif isinstance(node_data, dict) and feat in node_data:
                    node_feats.append(node_data[feat])
                else:
                    # Map feature names to actual node attributes
                    feat_value = 0.0
                    if feat == 'cpu_freq' and hasattr(node_data, 'free_cpu_freq'):
                        feat_value = node_data.free_cpu_freq
                    elif feat == 'buffer_size' and hasattr(node_data, 'task_buffer'):
                        feat_value = node_data.task_buffer.max_size if hasattr(node_data.task_buffer, 'max_size') else 0.0
                    elif feat == 'energy_level' and hasattr(node_data, 'energy_consumption'):
                        feat_value = 1.0 - min(1.0, node_data.energy_consumption / 100.0)  # Normalize energy
                    elif feat == 'task_count' and hasattr(node_data, 'active_tasks'):
                        feat_value = len(node_data.active_tasks)
                    
                    node_feats.append(feat_value)
            
            # Add temporal features if available
            if node in self.history_buffer:
                temporal_features = self._compute_temporal_features(node)
                node_feats.extend(temporal_features)
            else:
                # If no history, add zeros for temporal features
                node_feats.extend([0.0] * 5)  # Success rate, failure rate, etc.
            
            features.append(node_feats)
        
        # Convert to tensor
        if len(features) == 0:
            # Return empty tensor with correct shape
            return torch.zeros((0, len(self.feature_config['node_features']) + 5))
        
        return torch.tensor(features, dtype=torch.float)
    
    def extract_edge_features(self, graph: nx.Graph, edges: List[Tuple[int, int]]) -> torch.Tensor:
        """Extract edge features from the graph.
        
        Args:
            graph: NetworkX graph
            edges: List of edge tuples (source, target)
            
        Returns:
            Tensor of edge features
        """
        features = []
        
        for src, dst in edges:
            edge_data = graph.edges.get((src, dst), {})
            
            # Extract edge features
            edge_feats = []
            for feat in self.feature_config['edge_features']:
                if feat in edge_data:
                    edge_feats.append(edge_data[feat])
                else:
                    # Default value if feature not found
                    edge_feats.append(0.0)
            
            features.append(edge_feats)
        
        # Convert to tensor
        if len(features) == 0:
            # Return empty tensor with correct shape
            return torch.zeros((0, len(self.feature_config['edge_features'])))
        
        return torch.tensor(features, dtype=torch.float)
    
    def update_history(self, node_id: str, event_type: str, data: Dict):
        """Update the history buffer with new events.
        
        Args:
            node_id: ID of the node
            event_type: Type of event (task_success, task_failure, etc.)
            data: Event data
        """
        if node_id not in self.history_buffer:
            self.history_buffer[node_id] = []
        
        # Add timestamp to event data
        data['event_type'] = event_type
        data['timestamp'] = data.get('timestamp', pd.Timestamp.now())
        
        # Add to history buffer
        self.history_buffer[node_id].append(data)
        
        # Limit the history buffer size
        if len(self.history_buffer[node_id]) > self.feature_config['temporal_window']:
            self.history_buffer[node_id] = self.history_buffer[node_id][-self.feature_config['temporal_window']:]
    
    def _compute_temporal_features(self, node_id: str) -> List[float]:
        """Compute temporal features from the history buffer.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of temporal features
        """
        history = self.history_buffer.get(node_id, [])
        if not history:
            return [0.0] * 5
        
        # Compute success and failure rates
        events = pd.DataFrame(history)
        success_count = len(events[events['event_type'] == 'task_success'])
        failure_count = len(events[events['event_type'] == 'task_failure'])
        timeout_count = len(events[events['event_type'] == 'task_timeout'])
        total = max(1, len(events))
        
        success_rate = success_count / total
        failure_rate = failure_count / total
        timeout_rate = timeout_count / total
        
        # Compute trends (increasing or decreasing success rate)
        if len(events) > 1:
            # Split events in half and compare success rates
            half = len(events) // 2
            first_half = events.iloc[:half]
            second_half = events.iloc[half:]
            
            first_success = len(first_half[first_half['event_type'] == 'task_success']) / max(1, len(first_half))
            second_success = len(second_half[second_half['event_type'] == 'task_success']) / max(1, len(second_half))
            
            trend = second_success - first_success
        else:
            trend = 0.0
        
        # Compute recency-weighted success rate
        if len(events) > 0:
            weights = np.linspace(0.5, 1.0, len(events))  # More weight to recent events
            success_mask = (events['event_type'] == 'task_success').values.astype(float)
            recency_weighted = np.average(success_mask, weights=weights) if len(success_mask) > 0 else 0.0
        else:
            recency_weighted = 0.0
        
        return [success_rate, failure_rate, timeout_rate, trend, recency_weighted]
    
    def compute_graph_embeddings(self, graph: nx.Graph, method: str = 'node2vec', dim: int = None) -> Dict[str, np.ndarray]:
        """Compute graph embeddings for nodes.
        
        Args:
            graph: NetworkX graph
            method: Embedding method ('node2vec', 'spectral', etc.)
            dim: Embedding dimension
            
        Returns:
            Dictionary mapping node IDs to embeddings
        """
        if dim is None:
            dim = self.feature_config.get('embedding_dim', 16)
        
        # Use specified embedding method
        if method == 'node2vec':
            # Use node2vec from the RayCloudSim codebase
            try:
                from node2vec import Node2Vec
                node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=200, workers=4)
                model = node2vec.fit(window=10, min_count=1, batch_words=4)
                
                # Get embeddings for all nodes
                embeddings = {}
                for node in graph.nodes():
                    embeddings[node] = model.wv[str(node)]
                
                return embeddings
            except Exception as e:
                print(f"Warning: Node2Vec failed with error {e}. Falling back to spectral embedding.")
                method = 'spectral'
        
        if method == 'spectral':
            # Use spectral embedding
            from sklearn.manifold import SpectralEmbedding
            
            # Create adjacency matrix
            adj_matrix = nx.to_numpy_array(graph)
            
            # Compute spectral embedding
            embedding = SpectralEmbedding(n_components=dim, affinity='precomputed')
            node_embeddings = embedding.fit_transform(adj_matrix)
            
            # Map embeddings to node IDs
            embeddings = {node: node_embeddings[i] for i, node in enumerate(graph.nodes())}
            return embeddings
        
        # Fallback to random embeddings
        return {node: np.random.randn(dim) for node in graph.nodes()}

    def combine_features(self, node_feats: torch.Tensor, embeddings: Dict[str, np.ndarray], 
                        node_ids: List[str]) -> torch.Tensor:
        """Combine node features with graph embeddings.
        
        Args:
            node_feats: Tensor of node features
            embeddings: Dictionary mapping node IDs to embeddings
            node_ids: List of node IDs
            
        Returns:
            Combined feature tensor
        """
        combined = []
        
        for i, node_id in enumerate(node_ids):
            # Get node features
            if i < node_feats.shape[0]:
                feats = node_feats[i].numpy()
            else:
                feats = np.zeros(node_feats.shape[1])
            
            # Get node embedding
            if node_id in embeddings:
                emb = embeddings[node_id]
            else:
                # Use zeros if embedding not available
                emb = np.zeros(self.feature_config.get('embedding_dim', 16))
            
            # Combine features
            combined.append(np.concatenate([feats, emb]))
        
        # Convert to tensor
        if len(combined) == 0:
            # Return empty tensor with correct shape
            return torch.zeros((0, node_feats.shape[1] + self.feature_config.get('embedding_dim', 16)))
        
        # Convert list of arrays to single numpy array first, then to tensor
        combined_array = np.array(combined)
        return torch.tensor(combined_array, dtype=torch.float)