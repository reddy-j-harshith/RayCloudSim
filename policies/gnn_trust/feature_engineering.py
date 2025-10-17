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
            Tensor of node features [num_nodes, feature_dim]
        """
        features = []
        
        for node in nodes:
            # Get node data
            node_data = graph.nodes[node].get('data', {})
            if not isinstance(node_data, dict):
                node_data = {'node_id': node}
                
            # Extract static features
            node_feats = []
            
            # CPU utilization (normalized)
            if hasattr(node_data, 'free_cpu_freq') and hasattr(node_data, 'max_cpu_freq'):
                cpu_util = 1.0 - (node_data.free_cpu_freq / max(1.0, node_data.max_cpu_freq))
                node_feats.append(cpu_util)
            else:
                node_feats.append(0.5)  # default utilization
            
            # Buffer utilization (normalized)
            if hasattr(node_data, 'task_buffer'):
                buffer = node_data.task_buffer
                if hasattr(buffer, 'free_size') and hasattr(buffer, 'max_size'):
                    buffer_util = 1.0 - (buffer.free_size / max(1.0, buffer.max_size))
                    node_feats.append(buffer_util)
                else:
                    node_feats.append(0.5)
            else:
                node_feats.append(0.5)
            
            # Energy level (normalized)
            if hasattr(node_data, 'energy_consumption'):
                # Assume max consumption is 100
                energy_level = max(0.0, 1.0 - (node_data.energy_consumption / 100.0))
                node_feats.append(energy_level)
            elif hasattr(node_data, 'energy'):
                node_feats.append(min(1.0, max(0.0, node_data.energy)))
            else:
                node_feats.append(0.8)  # default high energy
            
            # Online status
            if hasattr(node_data, 'get_online'):
                online = 1.0 if node_data.get_online() else 0.0
                node_feats.append(online)
            else:
                node_feats.append(1.0)  # assume online
            
            # Task processing capability
            if hasattr(node_data, 'get_successful_tasks') and hasattr(node_data, 'get_total_tasks'):
                total_tasks = max(1, node_data.get_total_tasks())
                success_rate = node_data.get_successful_tasks() / total_tasks
                node_feats.append(success_rate)
            else:
                node_feats.append(0.5)  # default success rate
            
            # Current task load
            active_tasks = 0
            if hasattr(node_data, 'active_tasks'):
                active_tasks = len(node_data.active_tasks)
            elif hasattr(node_data, 'task_buffer') and hasattr(node_data.task_buffer, 'task_ids'):
                active_tasks = len(node_data.task_buffer.task_ids)
            
            # Normalize task load (assume max 10 tasks)
            task_load = min(1.0, active_tasks / 10.0)
            node_feats.append(task_load)
            
            # Network degree (centrality measure)
            degree = graph.degree(node) if node in graph else 0
            max_degree = max([graph.degree(n) for n in graph.nodes()]) if graph.nodes() else 1
            normalized_degree = degree / max(1.0, max_degree)
            node_feats.append(normalized_degree)
            
            # Clustering coefficient
            try:
                clustering = nx.clustering(graph, node) if node in graph else 0.0
                node_feats.append(clustering)
            except:
                node_feats.append(0.0)
            
            # Betweenness centrality (simplified)
            try:
                if len(graph.nodes()) > 2:
                    betweenness = nx.betweenness_centrality(graph).get(node, 0.0)
                    node_feats.append(betweenness)
                else:
                    node_feats.append(0.5)
            except:
                node_feats.append(0.5)
            
            # Add temporal features if available
            if node in self.history_buffer:
                temporal_features = self._compute_temporal_features(node)
                node_feats.extend(temporal_features)
            else:
                # If no history, add default temporal features
                node_feats.extend([0.5, 0.1, 0.05, 0.0, 0.5])  # success, failure, timeout, trend, recency
            
            # Add trust-related features
            trust_features = self._compute_trust_features(node, graph)
            node_feats.extend(trust_features)
            
            features.append(node_feats)
        
        # Convert to tensor
        if len(features) == 0:
            # Return empty tensor with correct shape
            feature_dim = 10 + 5 + 3  # static + temporal + trust features
            return torch.zeros((0, feature_dim))
        
        # Apply feature normalization
        features_tensor = torch.tensor(features, dtype=torch.float)
        features_tensor = self._normalize_features(features_tensor)
        
        return features_tensor
    
    def _compute_trust_features(self, node_id: str, graph: nx.Graph) -> List[float]:
        """Compute trust-related features for a node.
        
        Args:
            node_id: Node identifier
            graph: NetworkX graph
            
        Returns:
            List of trust-related features
        """
        trust_features = []
        
        # Average trust received from neighbors
        incoming_trust = []
        try:
            for neighbor in graph.neighbors(node_id):
                # Handle different graph types
                if graph.has_edge(neighbor, node_id):
                    if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
                        # For multigraphs
                        edge_data = {}
                        for key in graph[neighbor][node_id]:
                            edge_data = graph[neighbor][node_id][key]
                            break
                    else:
                        edge_data = graph[neighbor][node_id]
                    
                    trust_score = edge_data.get('trust', 0.5)
                    incoming_trust.append(trust_score)
        except:
            pass
        
        avg_incoming_trust = np.mean(incoming_trust) if incoming_trust else 0.5
        trust_features.append(avg_incoming_trust)
        
        # Trust variance (reputation stability)
        trust_variance = np.var(incoming_trust) if len(incoming_trust) > 1 else 0.0
        trust_features.append(trust_variance)
        
        # Trust consistency (how much this node's outgoing trust aligns with others)
        outgoing_trust = []
        try:
            for neighbor in graph.neighbors(node_id):
                if graph.has_edge(node_id, neighbor):
                    if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
                        # For multigraphs
                        edge_data = {}
                        for key in graph[node_id][neighbor]:
                            edge_data = graph[node_id][neighbor][key]
                            break
                    else:
                        edge_data = graph[node_id][neighbor]
                    
                    trust_score = edge_data.get('trust', 0.5)
                    outgoing_trust.append(trust_score)
        except:
            pass
        
        # Compare with network average trust
        all_trust_scores = []
        try:
            for _, _, edge_data in graph.edges(data=True):
                if isinstance(edge_data, dict) and 'trust' in edge_data:
                    all_trust_scores.append(edge_data['trust'])
        except:
            pass
        
        network_avg_trust = np.mean(all_trust_scores) if all_trust_scores else 0.5
        node_avg_trust = np.mean(outgoing_trust) if outgoing_trust else 0.5
        trust_alignment = 1.0 - abs(node_avg_trust - network_avg_trust)
        trust_features.append(trust_alignment)
        
        return trust_features
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features to improve training stability.
        
        Args:
            features: Feature tensor [num_nodes, feature_dim]
            
        Returns:
            Normalized feature tensor
        """
        if features.size(0) == 0:
            return features
        
        # Apply min-max normalization to ensure all features are in [0, 1]
        min_vals = features.min(dim=0, keepdim=True)[0]
        max_vals = features.max(dim=0, keepdim=True)[0]
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        
        normalized = (features - min_vals) / range_vals
        
        # Clamp to [0, 1] to handle any numerical issues
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        return normalized
    
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