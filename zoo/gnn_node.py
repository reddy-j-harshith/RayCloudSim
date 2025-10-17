"""GNN-based TrustNode implementation for RayCloudSim."""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

from zoo.node import TrustNode
from policies.gnn_trust.gnn_policy import GNNTrustPolicy
from core.infrastructure import Node

class GNNTrustNode(TrustNode):
    """TrustNode with GNN-based trust calculation."""
    
    def __init__(self, node_id, name, self_trust=0.5, max_cpu_freq=1e9, max_buffer_size=1e6,
                 location=None, idle_energy_coef=0, exe_energy_coef=0, is_malicious=False, gnn_config=None):
        """Initialize the GNN Trust Node.
        
        Args:
            node_id: Node ID
            name: Node name
            self_trust: Self trust value
            max_cpu_freq: Maximum CPU frequency
            max_buffer_size: Maximum buffer size
            location: Node location
            idle_energy_coef: Idle energy coefficient
            exe_energy_coef: Execution energy coefficient
            is_malicious: Whether the node is malicious
            gnn_config: Configuration for GNN-based trust
        """
        super().__init__(node_id, name, self_trust, max_cpu_freq, max_buffer_size,
                         location, idle_energy_coef, exe_energy_coef)
        
        # Store additional attributes
        self.is_malicious = is_malicious
        
        # Initialize GNN trust policy
        self.gnn_config = gnn_config or {}
        self.gnn_trust_policy = GNNTrustPolicy(self.gnn_config)
        self.infrastructure = None  # Will be set later
        
        # Add GNN-specific attributes
        self.embedding = None  # GNN embedding
        self.last_update_time = 0  # Last time embedding was updated
        self.update_interval = self.gnn_config.get('update_interval', 10)  # Update interval
        
    def set_infrastructure(self, infrastructure):
        """Set the infrastructure for the GNN trust policy.
        
        Args:
            infrastructure: The infrastructure object
        """
        self.infrastructure = infrastructure
        self.gnn_trust_policy.infrastructure = infrastructure
        
    def update_embedding(self, graph, force=False):
        """Update node embedding using the GNN.
        
        Args:
            graph: NetworkX graph
            force: Whether to force update even if not due
        """
        current_time = pd.Timestamp.now().timestamp()
        
        # Check if update is due
        if force or (current_time - self.last_update_time) >= self.update_interval:
            # Update the graph
            self.gnn_trust_policy._update_graph(graph, {self.name: self})
            
            # Update embeddings
            self.gnn_trust_policy._update_embeddings(graph)
            
            # Get updated embedding
            self.embedding = self.gnn_trust_policy.embedding_cache.get(self.name, None)
            
            # Update last update time
            self.last_update_time = current_time
    
    def compute_trust(self, node_id: str) -> float:
        """Compute trust score for a node.
        
        Args:
            node_id: ID of the node to compute trust for
            
        Returns:
            Trust score
        """
        # Use GNN-based trust score if available
        if node_id in self.gnn_trust_policy.trust_scores:
            return self.gnn_trust_policy.trust_scores[node_id]
        
        # Fall back to parent method
        return super().compute_trust(node_id)
    
    def task_completed(self, task, success: bool, flag: int):
        """Process task completion event.
        
        Args:
            task: Completed task
            success: Whether the task was successful
            flag: Flag indicating reason for success/failure
        """
        # Call parent method
        super().task_completed(task, success, flag)
        
        # Process feedback in GNN policy
        self.gnn_trust_policy.feedback(task, success, flag)
    
    def select_node(self, task, nodes: Dict[str, 'Node']) -> Optional[str]:
        """Select a node to offload the task to.
        
        Args:
            task: Task to offload
            nodes: Dictionary of available nodes
            
        Returns:
            Selected node name or None if no suitable node found
        """
        # Use GNN-based node selection
        return self.gnn_trust_policy.select(task, nodes)