#!/usr/bin/env python3

from research_attack_aware_system import ResearchAttackAwareSystem
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np

# Quick test with Pakistan Tuple30K and GAT only
print("Testing GNN model training directly...")
try:
    system = ResearchAttackAwareSystem(malicious_ratio=0.3)
    
    # Load dataset properly
    trainset, testset, dataset_info = system.load_dataset('pakistan', 'Tuple30K')
    print(f"Dataset info keys: {list(dataset_info.keys())}")
    
    # Just test with simple graph
    import networkx as nx
    network_graph = nx.complete_graph(8)  # Simple complete graph
    num_nodes = len(network_graph.nodes())
    
    # Test with minimal training data
    fake_data = {
        'final_trust_matrix': {i: {j: np.random.random() for j in range(num_nodes)} for i in range(num_nodes)},
        'phase_logger': type('obj', (object,), {'task_logs': [
            {'dst_node': i, 'execution_success': True, 'execution_time': 10, 'energy_consumed': 5} 
            for i in range(num_nodes)
        ]})(),
        'temporal_trust_data': []
    }
    
    gnn_results = system.train_gnn_models(fake_data, network_graph)
    
    if 'GAT' in gnn_results:
        print('SUCCESS: GAT model training works!')
        print(f'Train RMSE: {gnn_results["GAT"]["train_rmse"]:.4f}')
    else:
        print('FAILED: No GAT results')
        
except Exception as e:
    print(f'STILL FAILING: {e}')
    import traceback
    traceback.print_exc()