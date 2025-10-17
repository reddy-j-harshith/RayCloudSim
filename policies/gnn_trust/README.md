# GNN-based Trust Calculation for Task Offloading

This module implements a spatio-temporal Graph Neural Network (ST-GNN) framework for trust calculation in fog computing environments. It enables intelligent, attack-resilient task offloading decisions that maximize throughput while avoiding malicious or unreliable nodes.

## Overview

In fog computing environments, each node's behavior varies continuously with time due to changes in workload, connectivity, and external interference. This implementation models both:

1. Structural inter-node relations
2. Time-evolving behavioral dynamics

The framework computes dynamic trust representations for each fog node that reflect both recent performance history and position within the communication graph.

## Architecture

The GNN trust module consists of several key components:

### 1. GNN Architecture

- **Modular design** with configurable layers (GAT, GraphSAGE, GCN)
- **PyTorch integration** using PyTorch Geometric for efficient graph operations
- **Multiple model options** to evaluate performance across different architectures

### 2. Feature Engineering

- **Node features**: CPU frequency, buffer size, energy level, task count
- **Temporal features**: Success/failure rates, trends, recency-weighted metrics
- **Spatial embeddings**: Node2Vec or spectral embeddings preserving network structure
- **Edge features**: Latency, bandwidth, trust scores

### 3. Message Passing

- **Multi-hop propagation**: Information flow across multiple hops in the network
- **Edge-weighted communication**: Trust strength, latency, bandwidth as edge weights
- **Temporal integration**: Incorporating historical trust snapshots
- **Initialization strategy**: Proper handling of new nodes with limited history

### 4. Aggregation Strategies

- **Configurable aggregators**: Mean, sum, max, attention-weighted
- **Multi-head attention**: Learning to weigh neighbor messages based on relevance
- **Robust aggregation**: Detecting and handling outliers from malicious nodes

### 5. Contextual Thresholding

- **Dynamic thresholds**: Adapting to network conditions and task requirements
- **Context awareness**: Incorporating task criticality, congestion, attack frequency
- **QoS integration**: Stricter requirements for critical tasks
- **Attack responsiveness**: Automatic threshold elevation during attacks

## Integration with RayCloudSim

The GNN trust module integrates with RayCloudSim through:

1. The `GNNTrustPolicy` class that extends `BasePolicy`
2. The `GNNTrustNode` class that extends `TrustNode`

These classes enable seamless trust calculation and task offloading based on learned trust embeddings.

## Usage

```python
# Create a GNN-based trust node
gnn_node = GNNTrustNode(
    node_id="node1",
    name="n1",
    cpu_frequency=1e9,
    max_cpu_frequency=2e9,
    buffer_size=1e6,
    gnn_config={
        'model_type': 'gat',
        'hidden_dim': 64,
        'output_dim': 32
    }
)

# Use the node for task offloading
dst_name = gnn_node.select_node(task, available_nodes)
```

See `gnn_trust_demo.py` for a complete example.

## Configuration Options

The GNN trust module supports various configuration options:

### Model Types

- `gat`: Graph Attention Network
- `graphsage`: GraphSAGE
- `gcn`: Graph Convolutional Network

### Aggregation Methods

- `attention`: Attention-weighted aggregation
- `mean`: Mean aggregation
- `max`: Max aggregation
- `sum`: Sum aggregation
- `robust`: Outlier-resistant aggregation

### Other Parameters

- `hidden_dim`: Hidden dimension size
- `output_dim`: Output embedding dimension
- `num_layers`: Number of GNN layers
- `learning_rate`: Learning rate for model training
- `use_gpu`: Whether to use GPU acceleration
- `update_frequency`: How often to update the model
