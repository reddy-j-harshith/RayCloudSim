"""GNN-based trust policy for task offloading."""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import os
import pandas as pd
import time

from core.task import Task
from core.infrastructure import Node
from policies.base_policy import BasePolicy

from .models import GATModel, GraphSAGEModel, GCNModel
from .feature_engineering import FeatureExtractor
from .message_passing import MessagePassing
from .aggregation import AttentionAggregator, MeanAggregator, MaxAggregator, SumAggregator, RobustAggregator
from .threshold import ContextualThreshold

class GNNTrustPolicy(BasePolicy):
    """GNN-based trust policy for task offloading."""
    
    def __init__(self, config: Dict = None):
        """Initialize the policy.
        
        Args:
            config: Configuration for the policy
                - model_type: Type of GNN model ('gat', 'graphsage', 'gcn')
                - hidden_dim: Hidden dimension size
                - output_dim: Output embedding dimension
                - num_layers: Number of GNN layers
                - learning_rate: Learning rate for model training
                - use_gpu: Whether to use GPU acceleration
                - checkpoint_dir: Directory for model checkpoints
                - aggregation: Aggregation method ('attention', 'mean', 'max', 'sum', 'robust')
                - update_frequency: How often to update the model (in simulation steps)
                - importance_sampling: Whether to use importance sampling for training
        """
        super().__init__()
        
        default_config = {
            'model_type': 'gat',
            'hidden_dim': 64,
            'output_dim': 32,
            'num_layers': 2,
            'learning_rate': 0.001,
            'use_gpu': False,
            'checkpoint_dir': 'logs/gnn_trust',
            'aggregation': 'attention',
            'update_frequency': 10,
            'importance_sampling': True,
            'min_threshold': 0.2,
            'max_threshold': 0.6
        }
        
        # Merge with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.message_passing = MessagePassing()
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Initialize contextual threshold
        self.threshold = ContextualThreshold()
        
        # Initialize state variables
        self.current_step = 0
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['use_gpu'] else 'cpu')
        self.graph_cache = None
        self.embedding_cache = {}
        self.trust_scores = {}
        
        # Initialize model and optimizer
        self._init_model()
    
    def _init_model(self):
        """Initialize the GNN model based on configuration."""
        # Estimate input dimension based on features
        # This is a rough estimate, will be updated when actual features are available
        input_dim = len(self.feature_extractor.feature_config['node_features']) + 5 + self.feature_extractor.feature_config['embedding_dim']
        
        # Create model based on type
        if self.config['model_type'] == 'gat':
            self.model = GATModel(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim'],
                num_layers=self.config['num_layers']
            ).to(self.device)
        elif self.config['model_type'] == 'graphsage':
            self.model = GraphSAGEModel(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim'],
                num_layers=self.config['num_layers']
            ).to(self.device)
        else:  # default to GCN
            self.model = GCNModel(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim'],
                num_layers=self.config['num_layers']
            ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Initialize aggregator
        if self.config['aggregation'] == 'attention':
            self.aggregator = AttentionAggregator({'feature_dim': self.config['output_dim']})
        elif self.config['aggregation'] == 'max':
            self.aggregator = MaxAggregator()
        elif self.config['aggregation'] == 'sum':
            self.aggregator = SumAggregator()
        elif self.config['aggregation'] == 'robust':
            self.aggregator = RobustAggregator()
        else:  # default to mean
            self.aggregator = MeanAggregator()
    
    def act(self, env, task):
        """Act method required by BasePolicy.
        
        Args:
            env: Environment
            task: Task to process
            
        Returns:
            Selected node name or None
        """
        # Get available nodes from environment
        nodes = env.scenario.get_nodes()
        
        # Use the select method
        return self.select(task, nodes)
    
    def select(self, task: Task, nodes: Dict[str, Node]) -> Optional[str]:
        """Select a node to offload the task to.
        
        Args:
            task: Task to offload
            nodes: Dictionary of available nodes
            
        Returns:
            Selected node name or None if no suitable node found
        """
        # Update step counter
        self.current_step += 1
        
        # Check for empty nodes
        if not nodes:
            return None
        
        # Get networkx graph from infrastructure
        graph = self._get_graph()
        
        # Update graph with current node states
        self._update_graph(graph, nodes)
        
        # Update embeddings if needed
        if self.current_step % self.config['update_frequency'] == 0:
            self._update_embeddings(graph)
        
        # Calculate trust scores for each node
        self._calculate_trust_scores(graph, nodes.keys())
        
        # Get task context for threshold calculation
        task_context = {
            'criticality': getattr(task, 'criticality', 0.5),
            'qos_requirement': 1.0 if getattr(task, 'ddl', -1) > 0 else 0.5
        }
        
        # Get network state for threshold calculation
        network_state = {
            'congestion': self._estimate_congestion(nodes),
            'attack_detected': self._detect_attacks()
        }
        
        # Compute threshold based on context
        threshold = self.threshold.compute_threshold(task_context, network_state)
        
        # Filter nodes based on trust scores and resources
        candidates = []
        for name, node in nodes.items():
            # Check if node has enough resources for the task
            if not self._has_sufficient_resources(node, task):
                continue
            
            # Get trust score for this node
            trust_score = self.trust_scores.get(name, 0.0)
            
            # Check if trust score meets threshold
            if trust_score >= threshold:
                # Calculate a combined score based on trust and resources
                resource_score = self._calculate_resource_score(node, task)
                combined_score = 0.7 * trust_score + 0.3 * resource_score
                
                candidates.append((name, combined_score))
        
        # If no candidates meet the threshold, try relaxing it
        if not candidates and threshold > self.config['min_threshold']:
            relaxed_threshold = max(threshold * 0.8, self.config['min_threshold'])
            for name, node in nodes.items():
                # Check if node has enough resources for the task
                if not self._has_sufficient_resources(node, task):
                    continue
                
                # Get trust score for this node
                trust_score = self.trust_scores.get(name, 0.0)
                
                # Check if trust score meets relaxed threshold
                if trust_score >= relaxed_threshold:
                    # Calculate a combined score based on trust and resources
                    resource_score = self._calculate_resource_score(node, task)
                    combined_score = 0.7 * trust_score + 0.3 * resource_score
                    
                    candidates.append((name, combined_score))
        
        # If still no candidates, return None
        if not candidates:
            return None
        
        # Sort candidates by combined score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best candidate
        return candidates[0][0]
    
    def feedback(self, task: Task, success: bool, flag: int):
        """Process feedback about a completed task.
        
        Args:
            task: Completed task
            success: Whether the task was successful
            flag: Flag indicating reason for success/failure
        """
        # Get source and destination nodes
        src_name = task.src_name
        dst_name = task.dst_name
        
        # Skip if source or destination is None
        if not src_name or not dst_name:
            return
        
        # Create event data
        event_data = {
            'task_id': task.task_id,
            'task_size': task.task_size,
            'timestamp': time.time(),
            'success': success,
            'flag': flag
        }
        
        # Update feature extractor with this event
        event_type = 'task_success' if success else 'task_failure'
        if flag == 2:  # Timeout flag
            event_type = 'task_timeout'
        
        # Update history for destination node
        self.feature_extractor.update_history(dst_name, event_type, event_data)
        
        # Trigger immediate update for critical failures
        if not success and flag != 2:  # Not a timeout
            # Mark for update on next selection
            self.current_step = self.config['update_frequency'] - 1
    
    def train(self, graph: nx.Graph, labels: Dict[str, float] = None):
        """Train the GNN model with the current graph and optionally labels.
        
        Args:
            graph: NetworkX graph
            labels: Optional trust labels for nodes
        """
        # Skip if no model
        if self.model is None:
            return
        
        # Put model in training mode
        self.model.train()
        
        # Prepare features
        node_ids = list(graph.nodes())
        
        # Extract features
        node_features = self.feature_extractor.extract_node_features(graph, node_ids)
        
        # Compute spatial embeddings
        spatial_embeddings = self.feature_extractor.compute_graph_embeddings(graph)
        
        # Combine features
        combined_features = self.feature_extractor.combine_features(
            node_features, spatial_embeddings, node_ids
        ).to(self.device)
        
        # Prepare edge indices
        edge_list = list(graph.edges())
        edge_indices = torch.tensor([[node_ids.index(src), node_ids.index(dst)] 
                                     for src, dst in edge_list], dtype=torch.long).t().to(self.device)
        
        # Extract edge features if available
        edge_features = None
        if any('data' in edge_data for _, _, edge_data in graph.edges(data=True)):
            edge_features = self.feature_extractor.extract_edge_features(graph, edge_list).to(self.device)
        
        # Create target labels if available
        target = None
        if labels is not None:
            target = torch.tensor([labels.get(node, 0.0) for node in node_ids], dtype=torch.float).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(combined_features, edge_indices, edge_features)
        
        # Compute loss if targets available
        if target is not None:
            # Mean squared error loss
            loss = torch.nn.functional.mse_loss(output.squeeze(), target)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        
        return None
    
    def save_model(self, filename: Optional[str] = None):
        """Save the model to a file.
        
        Args:
            filename: Optional filename, otherwise use default
        """
        if self.model is None:
            return
        
        if filename is None:
            filename = os.path.join(self.config['checkpoint_dir'], f"{self.config['model_type']}_model.pt")
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filename)
    
    def load_model(self, filename: Optional[str] = None):
        """Load the model from a file.
        
        Args:
            filename: Optional filename, otherwise use default
        """
        if filename is None:
            filename = os.path.join(self.config['checkpoint_dir'], f"{self.config['model_type']}_model.pt")
        
        # Check if file exists
        if not os.path.exists(filename):
            return
        
        # Load model state
        checkpoint = torch.load(filename, map_location=self.device)
        
        # Update config if available
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # Re-initialize model with loaded config
        self._init_model()
        
        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def _get_graph(self) -> nx.Graph:
        """Get the NetworkX graph from the infrastructure.
        
        Returns:
            NetworkX graph
        """
        # Use cached graph if available
        if self.graph_cache is not None:
            return self.graph_cache
        
        # Create a new graph
        graph = nx.DiGraph()
        
        # Get infrastructure if available
        if hasattr(self, 'infrastructure') and self.infrastructure:
            # Add all nodes to the graph
            for node_name, node in self.infrastructure.get_nodes().items():
                graph.add_node(node_name, node=node)
            
            # Add all links as edges to the graph
            links = self.infrastructure.get_links()
            
            if isinstance(links, dict):
                # Links are returned as dictionary with tuple keys or string keys
                for link_key, link in links.items():
                    if isinstance(link_key, tuple) and len(link_key) >= 2:
                        # Key is tuple (src, dst, idx)
                        src, dst = link_key[0], link_key[1]
                        graph.add_edge(src, dst, link=link)
                    elif isinstance(link_key, str) and "->" in link_key:
                        # Key is string "src->dst"
                        src, dst = link_key.split("->")
                        graph.add_edge(src, dst, link=link)
            elif isinstance(links, list):
                # Links are returned as list of tuples (src, dst, idx)
                for link_info in links:
                    if len(link_info) >= 2:
                        src, dst = link_info[0], link_info[1]
                        # Add edge (we'll get the link object later if needed)
                        graph.add_edge(src, dst)
        
        # Cache the graph
        self.graph_cache = graph
        
        return graph
    
    def _update_graph(self, graph: nx.Graph, nodes: Dict[str, Node]):
        """Update the graph with current node states.
        
        Args:
            graph: NetworkX graph
            nodes: Dictionary of nodes
        """
        # Add nodes to graph if not already present
        for name, node in nodes.items():
            if name not in graph:
                graph.add_node(name, data=node)
            else:
                # Update node data
                graph.nodes[name]['data'] = node
        
        # Add edges based on connectivity
        for src_name, src_node in nodes.items():
            for dst_name, dst_node in nodes.items():
                if src_name != dst_name:
                    # Check if nodes are connected
                    # This is a simplified check, actual connectivity would depend on infrastructure
                    if not graph.has_edge(src_name, dst_name):
                        # Add edge with default attributes
                        graph.add_edge(src_name, dst_name, weight=1.0, trust=0.5)
    
    def _update_embeddings(self, graph: nx.Graph):
        """Update node embeddings using the GNN model.
        
        Args:
            graph: NetworkX graph
        """
        # Skip if no model or empty graph
        if self.model is None or len(graph) == 0:
            return
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Prepare features
        node_ids = list(graph.nodes())
        
        # Extract features
        node_features = self.feature_extractor.extract_node_features(graph, node_ids)
        
        # Compute spatial embeddings
        spatial_embeddings = self.feature_extractor.compute_graph_embeddings(graph)
        
        # Combine features
        combined_features = self.feature_extractor.combine_features(
            node_features, spatial_embeddings, node_ids
        ).to(self.device)
        
        # Prepare edge indices
        edge_list = list(graph.edges())
        if not edge_list:
            # If no edges, create self-loops
            edge_list = [(node, node) for node in node_ids]
        
        edge_indices = torch.tensor([[node_ids.index(src), node_ids.index(dst)] 
                                     for src, dst in edge_list], dtype=torch.long).t().to(self.device)
        
        # Extract edge features if available
        edge_features = None
        if any('data' in edge_data for _, _, edge_data in graph.edges(data=True)):
            edge_features = self.feature_extractor.extract_edge_features(graph, edge_list).to(self.device)
        
        # Forward pass to get embeddings
        with torch.no_grad():
            embeddings = self.model(combined_features, edge_indices, edge_features)
        
        # Update embedding cache
        for i, node_id in enumerate(node_ids):
            self.embedding_cache[node_id] = embeddings[i].cpu().numpy()
    
    def _calculate_trust_scores(self, graph: nx.Graph, node_names: List[str]):
        """Calculate trust scores for nodes based on embeddings.
        
        Args:
            graph: NetworkX graph
            node_names: List of node names to calculate scores for
        """
        # Skip if no embeddings
        if not self.embedding_cache:
            # Initialize with default scores
            for name in node_names:
                self.trust_scores[name] = 0.5
            return
        
        # Calculate trust scores for each node
        for name in node_names:
            if name in self.embedding_cache:
                # Get node embedding
                embedding = self.embedding_cache[name]
                
                # Calculate trust score (normalize to [0, 1])
                # Use cosine similarity with a reference "good" embedding
                # This is a simplified approach; in practice would use more sophisticated scoring
                trust_score = (embedding.mean() + 1) / 2
                
                # Clamp to valid range
                trust_score = max(0.0, min(1.0, trust_score))
                
                # Update trust score
                self.trust_scores[name] = trust_score
            else:
                # No embedding available, use default
                self.trust_scores[name] = 0.5
    
    def _has_sufficient_resources(self, node: Node, task: Task) -> bool:
        """Check if a node has sufficient resources for a task.
        
        Args:
            node: Node to check
            task: Task to execute
            
        Returns:
            True if node has sufficient resources, False otherwise
        """
        # Check if node has sufficient buffer
        if hasattr(node, 'task_buffer') and hasattr(node.task_buffer, 'free_size'):
            if node.task_buffer.free_size < task.task_size:
                return False
        
        # Check if node has sufficient CPU
        if hasattr(node, 'free_cpu_freq'):
            required_cpu = task.cycles_per_bit * task.task_size
            if node.free_cpu_freq < required_cpu:
                return False
        
        return True
    
    def _calculate_resource_score(self, node: Node, task: Task) -> float:
        """Calculate a resource availability score for a node.
        
        Args:
            node: Node to calculate score for
            task: Task to execute
            
        Returns:
            Resource score [0, 1]
        """
        scores = []
        
        # Score buffer availability
        if hasattr(node, 'task_buffer') and hasattr(node.task_buffer, 'free_size') and hasattr(node.task_buffer, 'max_size'):
            buffer_ratio = node.task_buffer.free_size / max(1, node.task_buffer.max_size)
            scores.append(buffer_ratio)
        
        # Score CPU availability
        if hasattr(node, 'free_cpu_freq') and hasattr(node, 'max_cpu_freq'):
            cpu_ratio = node.free_cpu_freq / max(1, node.max_cpu_freq)
            scores.append(cpu_ratio)
        
        # Score energy level
        if hasattr(node, 'energy'):
            # Assuming energy is normalized to [0, 1]
            scores.append(node.energy)
        
        # Return average score
        if scores:
            return sum(scores) / len(scores)
        
        # Default score
        return 0.5
    
    def _estimate_congestion(self, nodes: Dict[str, Node]) -> float:
        """Estimate the current network congestion level.
        
        Args:
            nodes: Dictionary of nodes
            
        Returns:
            Congestion level [0, 1]
        """
        # Calculate average buffer utilization
        buffer_utils = []
        for name, node in nodes.items():
            if hasattr(node, 'task_buffer') and hasattr(node.task_buffer, 'free_size') and hasattr(node.task_buffer, 'max_size'):
                buffer_util = 1 - (node.task_buffer.free_size / max(1, node.task_buffer.max_size))
                buffer_utils.append(buffer_util)
        
        # Return average utilization
        if buffer_utils:
            return sum(buffer_utils) / len(buffer_utils)
        
        # Default level
        return 0.0
    
    def _detect_attacks(self) -> float:
        """Detect potential attacks in the network.
        
        Returns:
            Attack detection level [0, 1]
        """
        # Analyze failure patterns in recent history
        failure_counts = {}
        total_events = {}
        
        for node_id, history in self.feature_extractor.history_buffer.items():
            # Count failures and total events
            failures = sum(1 for event in history if event['event_type'] == 'task_failure')
            events = len(history)
            
            failure_counts[node_id] = failures
            total_events[node_id] = events
        
        # Calculate failure rates
        failure_rates = []
        for node_id in failure_counts:
            if total_events[node_id] > 0:
                failure_rate = failure_counts[node_id] / total_events[node_id]
                failure_rates.append(failure_rate)
        
        # Check for high failure rates
        if failure_rates:
            # Consider attack if any node has high failure rate (>50%)
            max_failure_rate = max(failure_rates)
            return max(0.0, min(1.0, max_failure_rate * 2 - 0.5))  # Scale to [0, 1]
        
        # Default level
        return 0.0