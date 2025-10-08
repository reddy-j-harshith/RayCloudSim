"""
Simple test script to verify GNN model improvements.
"""

import os
import sys
import numpy as np
import torch
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.env import Env_Trust
from examples.scenarios.trust_scenario_1 import Scenario

from policies.gnn_trust.training import TrustTrainer
from policies.gnn_trust.feature_engineering import FeatureExtractor

def create_simple_test_data():
    """Create simple test data for GNN models."""
    print("Creating simple test data...")
    
    # Create a simple graph
    graph = nx.Graph()
    
    # Add nodes
    nodes = ['n0', 'n1', 'n2', 'n3', 'n4']
    for i, node in enumerate(nodes):
        # Create mock node data
        node_data = type('Node', (), {
            'free_cpu_freq': 50.0 + np.random.normal(0, 10),
            'max_cpu_freq': 100.0,
            'task_buffer': type('Buffer', (), {
                'free_size': 80.0 + np.random.normal(0, 15),
                'max_size': 100.0,
                'task_ids': []
            })(),
            'energy': 0.8 + np.random.normal(0, 0.1),
            'active_tasks': [],
            'is_malicious': i in [2, 4],  # Make nodes 2 and 4 malicious
            'get_online': lambda: True,
            'get_successful_tasks': lambda: 8 + np.random.randint(-2, 3),
            'get_total_tasks': lambda: 10
        })()
        
        graph.add_node(node, data=node_data)
    
    # Add edges (fully connected)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            graph.add_edge(nodes[i], nodes[j])
    
    # Create trust labels (malicious nodes have lower trust)
    trust_labels = {}
    for i, node in enumerate(nodes):
        if i in [2, 4]:  # Malicious nodes
            trust_labels[node] = np.random.uniform(0.1, 0.4)
        else:  # Honest nodes
            trust_labels[node] = np.random.uniform(0.6, 0.9)
    
    # Add trust scores to edges
    for src, dst in graph.edges():
        dst_trust = trust_labels.get(dst, 0.5)
        edge_trust = dst_trust + np.random.normal(0, 0.05)
        edge_trust = max(0.0, min(1.0, edge_trust))
        graph[src][dst]['trust'] = edge_trust
    
    return [graph], [trust_labels]

def test_models():
    """Test all GNN models."""
    print("🧪 Testing GNN Models")
    print("=" * 50)
    
    # Create test data
    graphs, trust_labels = create_simple_test_data()
    
    # Test each model type
    models = ['gat', 'graphsage', 'gcn']
    results = {}
    
    for model_type in models:
        print(f"\nTesting {model_type.upper()} model...")
        
        try:
            # Create trainer with smaller config for testing
            config = {
                'hidden_dim': 32,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.01,
                'num_epochs': 50,
                'early_stopping_patience': 10,
                'val_split': 0.0,  # No validation for this simple test
                'test_split': 0.0,
                'trust_threshold': 0.5,
                'log_interval': 25
            }
            
            trainer = TrustTrainer(model_type=model_type, config=config)
            
            # Get input dimension
            feature_extractor = FeatureExtractor()
            node_ids = list(graphs[0].nodes())
            node_features = feature_extractor.extract_node_features(graphs[0], node_ids)
            spatial_embeddings = feature_extractor.compute_graph_embeddings(graphs[0])
            combined_features = feature_extractor.combine_features(node_features, spatial_embeddings, node_ids)
            input_dim = combined_features.shape[1]
            
            print(f"  Input dimension: {input_dim}")
            
            # Create model
            trainer.create_model(input_dim)
            
            # Train on single graph (overfitting test)
            train_loader, _, _ = trainer.prepare_datasets(graphs * 20, trust_labels * 20)  # Repeat data
            
            print("  Training...")
            trainer.train(train_loader, None)
            
            # Test prediction
            predictions = trainer.predict(graphs[0])
            true_labels = trust_labels[0]
            
            # Calculate metrics
            y_true = [1 if true_labels[node] > 0.5 else 0 for node in node_ids]
            y_pred = [1 if predictions[node] > 0.5 else 0 for node in node_ids]
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results[model_type] = {
                'accuracy': accuracy,
                'f1': f1,
                'predictions': predictions,
                'true_labels': true_labels
            }
            
            print(f"  ✅ Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
            # Print detailed predictions
            print("  Predictions vs Truth:")
            for node in node_ids:
                pred = predictions[node]
                true = true_labels[node]
                malicious = "🔴" if node in ['n2', 'n4'] else "🟢"
                print(f"    {node}: {pred:.3f} (true: {true:.3f}) {malicious}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[model_type] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*50}")
    
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{model_type.upper()}: ❌ {result['error']}")
        else:
            print(f"{model_type.upper()}: ✅ Acc={result['accuracy']:.3f}, F1={result['f1']:.3f}")
    
    # Check if models learned to distinguish malicious nodes
    print(f"\n🎯 MALICIOUS NODE DETECTION:")
    malicious_nodes = ['n2', 'n4']
    honest_nodes = ['n0', 'n1', 'n3']
    
    for model_type, result in results.items():
        if 'predictions' in result:
            predictions = result['predictions']
            
            # Average trust for malicious vs honest
            mal_trust = np.mean([predictions[node] for node in malicious_nodes])
            hon_trust = np.mean([predictions[node] for node in honest_nodes])
            
            separation = hon_trust - mal_trust
            print(f"  {model_type.upper()}: Honest={hon_trust:.3f}, Malicious={mal_trust:.3f}, Gap={separation:.3f}")
            
            if separation > 0.1:
                print(f"    ✅ Good separation!")
            else:
                print(f"    ⚠️ Poor separation")

def main():
    print("🚀 Simple GNN Model Test")
    print("=" * 50)
    
    # Test models
    test_models()
    
    print("\n✅ Testing completed!")

if __name__ == '__main__':
    main()