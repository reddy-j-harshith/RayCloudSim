"""
Simple test script to validate the GNN trust implementation.
"""
import os
import sys
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from policies.gnn_trust.models import GATModel
from policies.gnn_trust.feature_engineering import FeatureExtractor
from policies.gnn_trust.message_passing import MessagePassing
from policies.gnn_trust.aggregation import AttentionAggregator, MeanAggregator
from policies.gnn_trust.threshold import ContextualThreshold

def test_gnn_model():
    """Test GNN model."""
    print("Testing GNN model...")
    
    # Create a simple graph
    graph = nx.Graph()
    graph.add_nodes_from(['n0', 'n1', 'n2', 'n3', 'n4'])
    graph.add_edges_from([('n0', 'n1'), ('n1', 'n2'), ('n2', 'n3'), ('n3', 'n4'), ('n4', 'n0')])
    
    # Add node attributes
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['data'] = {
            'node_id': node,
            'cpu_freq': 1e9 * (i + 1),
            'buffer_size': 1e6 * (i + 1),
            'energy_level': 0.8 - i * 0.1,
            'task_count': i
        }
    
    # Add edge attributes
    for i, (src, dst) in enumerate(graph.edges()):
        graph.edges[src, dst]['latency'] = 0.1 * (i + 1)
        graph.edges[src, dst]['bandwidth'] = 1e6 / (i + 1)
        graph.edges[src, dst]['trust_score'] = 0.9 - i * 0.1
    
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Extract node features
    node_features = feature_extractor.extract_node_features(graph, list(graph.nodes()))
    print(f"Node features shape: {node_features.shape}")
    
    # Extract edge features
    edge_features = feature_extractor.extract_edge_features(graph, list(graph.edges()))
    print(f"Edge features shape: {edge_features.shape}")
    
    # Compute graph embeddings
    embeddings = feature_extractor.compute_graph_embeddings(graph, method='spectral', dim=8)
    print(f"Graph embeddings: {len(embeddings)} nodes with dimension {len(next(iter(embeddings.values())))}") 
    
    # Combine features
    combined_features = feature_extractor.combine_features(node_features, embeddings, list(graph.nodes()))
    print(f"Combined features shape: {combined_features.shape}")
    
    # Create edge index (proper format for PyTorch Geometric)
    node_list = list(graph.nodes())
    edge_list = list(graph.edges())
    edge_index = torch.tensor([[node_list.index(src), node_list.index(dst)] for src, dst in edge_list], 
                              dtype=torch.long).t()
    print(f"Edge index shape: {edge_index.shape}")
    
    # Create GNN model
    model = GATModel(
        input_dim=combined_features.shape[1],
        hidden_dim=32,
        output_dim=16,
        num_layers=2
    )
    
    # Forward pass
    output = model(combined_features, edge_index, edge_features)
    print(f"Model output shape: {output.shape}")
    
    # Test message passing
    message_passing = MessagePassing()
    updated_features = message_passing.propagate_trust(combined_features, edge_index, edge_features)
    print(f"Updated features shape: {updated_features.shape}")
    
    # Test aggregation
    aggregator = AttentionAggregator({'feature_dim': combined_features.shape[1]})
    aggregated = aggregator.aggregate(combined_features)
    print(f"Aggregated features shape: {aggregated.shape}")
    
    # Test threshold
    threshold = ContextualThreshold()
    task_context = {'criticality': 0.8, 'qos_requirement': 0.7}
    network_state = {'congestion': 0.3, 'attack_detected': 0.1}
    computed_threshold = threshold.compute_threshold(task_context, network_state)
    print(f"Computed threshold: {computed_threshold}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_gnn_model()