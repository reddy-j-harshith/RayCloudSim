"""Simplified and robust feature engineering for GNN trust calculation."""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def extract_node_features(G, node_embeddings=None, embedding_dim=16):
    """
    Extract comprehensive features for each node in the graph.
    
    Args:
        G: NetworkX graph
        node_embeddings: Optional pre-computed node embeddings
        embedding_dim: Dimension of node embeddings if computing new ones
    
    Returns:
        torch.Tensor: Node features matrix [num_nodes, num_features]
    """
    if len(G.nodes()) == 0:
        return torch.zeros((0, 33))  # 18 + 16 - 1 features
    
    num_nodes = len(G.nodes())
    node_list = list(G.nodes())
    
    # Initialize feature matrix: 17 basic features + 16 embedding features
    features = torch.zeros((num_nodes, 33))
    
    try:
        # Node2Vec embeddings with robust error handling
        embedding_success = False
        if node_embeddings is None:
            try:
                # Convert to simple graph for Node2Vec if needed
                if G.is_multigraph():
                    simple_G = nx.Graph()
                    simple_G.add_nodes_from(G.nodes(data=True))
                    # Add edges without duplicates
                    edges_added = set()
                    for u, v, data in G.edges(data=True):
                        if (u, v) not in edges_added and (v, u) not in edges_added:
                            simple_G.add_edge(u, v, **data)
                            edges_added.add((u, v))
                    G_for_embedding = simple_G
                else:
                    G_for_embedding = G
                
                if len(G_for_embedding.edges()) > 0:
                    embeddings = _compute_node2vec_embeddings(G_for_embedding, embedding_dim)
                    if embeddings is not None and len(embeddings) > 0:
                        for i, node in enumerate(node_list):
                            if node in embeddings:
                                features[i, 17:] = torch.tensor(embeddings[node], dtype=torch.float32)
                                embedding_success = True
            except Exception as e:
                print(f"Node2Vec embedding failed: {str(e)[:100]}")
                
        else:
            try:
                for i, node in enumerate(node_list):
                    if node in node_embeddings:
                        embedding = node_embeddings[node]
                        if len(embedding) == embedding_dim:
                            features[i, 17:] = torch.tensor(embedding, dtype=torch.float32)
                            embedding_success = True
                        else:
                            features[i, 17:] = torch.randn(embedding_dim) * 0.1
                    else:
                        features[i, 17:] = torch.randn(embedding_dim) * 0.1
            except:
                pass
        
        # If embeddings failed, use random embeddings
        if not embedding_success:
            features[:, 17:] = torch.randn(num_nodes, embedding_dim) * 0.1
        
        # Basic graph features
        degree_dict = dict(G.degree())
        
        # Compute clustering safely
        try:
            clustering_dict = nx.clustering(G)
        except:
            clustering_dict = {node: 0.0 for node in G.nodes()}
        
        # Compute centrality measures safely
        betweenness_dict = {}
        closeness_dict = {}
        eigenvector_dict = {}
        
        try:
            if len(G.edges()) > 0:
                # Convert to simple graph for centrality if needed
                G_centrality = G
                if G.is_multigraph():
                    G_centrality = nx.Graph()
                    G_centrality.add_nodes_from(G.nodes())
                    G_centrality.add_edges_from(G.edges())
                
                betweenness_dict = nx.betweenness_centrality(G_centrality)
                closeness_dict = nx.closeness_centrality(G_centrality)
                eigenvector_dict = nx.eigenvector_centrality(G_centrality, max_iter=100)
            else:
                betweenness_dict = {node: 0.0 for node in G.nodes()}
                closeness_dict = {node: 0.0 for node in G.nodes()}
                eigenvector_dict = {node: 0.0 for node in G.nodes()}
        except:
            betweenness_dict = {node: 0.0 for node in G.nodes()}
            closeness_dict = {node: 0.0 for node in G.nodes()}
            eigenvector_dict = {node: 0.0 for node in G.nodes()}
        
        # Extract features for each node
        for i, node in enumerate(node_list):
            node_data = G.nodes[node] if node in G.nodes() else {}
            
            # Basic features
            features[i, 0] = degree_dict.get(node, 0)
            features[i, 1] = clustering_dict.get(node, 0)
            features[i, 2] = betweenness_dict.get(node, 0)
            features[i, 3] = closeness_dict.get(node, 0)
            features[i, 4] = eigenvector_dict.get(node, 0)
            
            # Node attributes (with safe defaults)
            features[i, 5] = _safe_get_attr(node_data, 'cpu_utilization', 0.5)
            features[i, 6] = _safe_get_attr(node_data, 'memory_usage', 0.5)
            features[i, 7] = _safe_get_attr(node_data, 'energy_level', 1.0)
            features[i, 8] = _safe_get_attr(node_data, 'buffer_size', 100.0) / 1000.0  # Normalize
            features[i, 9] = _safe_get_attr(node_data, 'network_load', 0.3)
            
            # Trust-related features
            trust_features = _compute_trust_features(G, node)
            features[i, 10:17] = torch.tensor(trust_features, dtype=torch.float32)
        
        # Normalize features
        features = _normalize_features(features)
        
        return features
        
    except Exception as e:
        print(f"Feature extraction failed: {str(e)[:100]}")
        # Return safe default features
        features = torch.zeros(num_nodes, 33)
        for i in range(num_nodes):
            features[i, 0] = 1.0  # Basic degree
            features[i, 5:10] = 0.5  # Basic node attributes
            features[i, 17:] = torch.randn(embedding_dim) * 0.1  # Random embeddings
        return features

def _safe_get_attr(node_data, attr_name, default_value):
    """Safely get attribute from node data."""
    try:
        if isinstance(node_data, dict):
            return node_data.get(attr_name, default_value)
        elif hasattr(node_data, attr_name):
            value = getattr(node_data, attr_name)
            if callable(value):
                return value()
            return value
        else:
            return default_value
    except:
        return default_value

def _compute_node2vec_embeddings(G, embedding_dim=16):
    """Compute Node2Vec embeddings for graph nodes."""
    try:
        from node2vec import Node2Vec
        
        # Create Node2Vec model with reduced parameters for stability
        node2vec = Node2Vec(
            G, 
            dimensions=embedding_dim,
            walk_length=10,  # Reduced for stability
            num_walks=50,    # Reduced for speed
            workers=1,       # Single worker to avoid issues
            p=1, 
            q=1
        )
        
        # Train model
        model = node2vec.fit(
            window=5,        # Reduced window
            min_count=1,
            batch_words=4,
            epochs=1,        # Single epoch for speed
            sg=1,            # Skip-gram
            hs=0,            # Use negative sampling
            negative=5,      # Negative sampling
            workers=1        # Single worker
        )
        
        # Extract embeddings
        embeddings = {}
        for node in G.nodes():
            try:
                embeddings[node] = model.wv[str(node)]
            except:
                embeddings[node] = np.random.randn(embedding_dim) * 0.1
        
        return embeddings
        
    except Exception as e:
        print(f"Node2Vec failed: {str(e)[:50]}...")
        return None

def _compute_trust_features(G, node):
    """Compute trust-related features for a node."""
    try:
        trust_features = []
        
        # Average trust received from neighbors
        incoming_trust = []
        outgoing_trust = []
        
        try:
            for neighbor in G.neighbors(node):
                # Incoming trust
                if G.has_edge(neighbor, node):
                    edge_data = _get_edge_data(G, neighbor, node)
                    trust_score = edge_data.get('trust', 0.5)
                    incoming_trust.append(trust_score)
                
                # Outgoing trust
                if G.has_edge(node, neighbor):
                    edge_data = _get_edge_data(G, node, neighbor)
                    trust_score = edge_data.get('trust', 0.5)
                    outgoing_trust.append(trust_score)
        except:
            pass
        
        # Average incoming trust
        avg_incoming_trust = np.mean(incoming_trust) if incoming_trust else 0.5
        trust_features.append(avg_incoming_trust)
        
        # Trust variance (reputation stability)
        trust_variance = np.var(incoming_trust) if len(incoming_trust) > 1 else 0.0
        trust_features.append(trust_variance)
        
        # Average outgoing trust
        avg_outgoing_trust = np.mean(outgoing_trust) if outgoing_trust else 0.5
        trust_features.append(avg_outgoing_trust)
        
        # Trust consistency (how consistent are the trust values)
        all_trust = incoming_trust + outgoing_trust
        trust_consistency = 1.0 - (np.var(all_trust) if len(all_trust) > 1 else 0.0)
        trust_features.append(trust_consistency)
        
        # Trust ratio (incoming vs outgoing)
        if avg_outgoing_trust > 0:
            trust_ratio = avg_incoming_trust / avg_outgoing_trust
        else:
            trust_ratio = 1.0
        trust_features.append(min(2.0, trust_ratio))  # Cap at 2.0
        
        # Number of trust relationships
        num_trust_relationships = len(incoming_trust) + len(outgoing_trust)
        normalized_relationships = min(1.0, num_trust_relationships / 10.0)
        trust_features.append(normalized_relationships)
        
        # Trust diversity (how many different trust levels)
        unique_trust_levels = len(set(all_trust)) if all_trust else 1
        trust_diversity = min(1.0, unique_trust_levels / 5.0)
        trust_features.append(trust_diversity)
        
        return trust_features
        
    except Exception as e:
        # Return default trust features
        return [0.5, 0.0, 0.5, 1.0, 1.0, 0.1, 0.2]

def _get_edge_data(G, u, v):
    """Safely get edge data from graph."""
    try:
        if G.is_multigraph():
            # For multigraphs, get the first edge
            if G.has_edge(u, v):
                for key in G[u][v]:
                    return G[u][v][key]
            return {}
        else:
            return G[u][v] if G.has_edge(u, v) else {}
    except:
        return {}

def _normalize_features(features):
    """Normalize features to improve training stability."""
    try:
        if features.size(0) == 0:
            return features
        
        # Apply min-max normalization column-wise
        min_vals = features.min(dim=0, keepdim=True)[0]
        max_vals = features.max(dim=0, keepdim=True)[0]
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        
        normalized = (features - min_vals) / range_vals
        
        # Clamp to [0, 1] to handle numerical issues
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        # Replace NaN values with 0.5
        normalized = torch.where(torch.isnan(normalized), 
                                torch.full_like(normalized, 0.5), 
                                normalized)
        
        return normalized
        
    except Exception as e:
        print(f"Normalization failed: {str(e)[:50]}")
        # Return original features if normalization fails
        return torch.clamp(features, 0.0, 1.0)