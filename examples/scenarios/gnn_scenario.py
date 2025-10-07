"""Scenario for GNN-based Trust Evaluation."""

import os
import sys
import json
from core.base_scenario import BaseScenario
from zoo.node import TrustNode, MaliciousNode
from zoo.gnn_node import GNNTrustNode
from core.infrastructure import Node, Location

class Scenario(BaseScenario):
    """Scenario implementation for GNN-based trust evaluation."""

    def load_config(self, config_file: str) -> dict:
        """Load configuration file.
        
        Args:
            config_file: Path to config file
            
        Returns:
            JSON object from config file
        """
        with open(config_file, 'r') as fr:
            return json.load(fr)

    def init_infrastructure_nodes(self):
        """Initialize infrastructure nodes."""
        for node_info in self.json_nodes:
            # Get node info
            node_id = node_info["NodeId"]
            name = node_info["NodeName"]
            location = self.get_location(node_info)
            
            # Get node type
            node_type = node_info.get("NodeType", "Node")
            
            # Create node based on type
            if node_type == "GNNTrustNode":
                node = GNNTrustNode(
                    node_id=node_id,
                    name=name,
                    self_trust=0.8,  # Default self trust
                    max_cpu_freq=node_info.get("MaxCpuFreq", 1000),
                    max_buffer_size=node_info.get("MaxBufferSize", 1000),
                    location=location,
                    idle_energy_coef=node_info.get("IdleEnergyCoef", 0.01),
                    exe_energy_coef=node_info.get("ExeEnergyCoef", 1.0),
                    gnn_config={
                        'model_type': 'gat',
                        'hidden_dim': 64,
                        'output_dim': 32,
                        'update_frequency': 5
                    }
                )
            elif node_type == "TrustNode":
                node = TrustNode(
                    node_id=node_id,
                    name=name,
                    self_trust=0.8,  # Default self trust
                    max_cpu_freq=node_info.get("MaxCpuFreq", 1000),
                    max_buffer_size=node_info.get("MaxBufferSize", 1000),
                    location=location,
                    idle_energy_coef=node_info.get("IdleEnergyCoef", 0.01),
                    exe_energy_coef=node_info.get("ExeEnergyCoef", 1.0)
                )
            elif node_type == "MaliciousNode":
                node = MaliciousNode(
                    node_id=node_id,
                    name=name,
                    self_trust=0.2,  # Low self trust for malicious nodes
                    mal_type=1,  # Default malicious type
                    max_cpu_freq=node_info.get("MaxCpuFreq", 1000),
                    max_buffer_size=node_info.get("MaxBufferSize", 1000),
                    location=location,
                    idle_energy_coef=node_info.get("IdleEnergyCoef", 0.01),
                    exe_energy_coef=node_info.get("ExeEnergyCoef", 1.0)
                )
            else:
                node = Node(
                    node_id=node_id,
                    name=name
                )
                node.cpu_freq = node_info.get("cpu_freq", 1e9)
                node.max_cpu_freq = node_info.get("max_cpu_freq", 1e9)
                if "buffer_size" in node_info:
                    node.init_buffer(node_info["buffer_size"])
                if location:
                    node.location = location
            
            # Add node to infrastructure
            self.infrastructure.add_node(node)
            self.node_id2name[node_id] = name

    def get_location(self, node_info: dict) -> Location:
        """Get location from node info.
        
        Args:
            node_info: Node information dictionary
            
        Returns:
            Location object or None
        """
        if "LocX" in node_info and "LocY" in node_info:
            return Location(node_info["LocX"], node_info["LocY"])
        elif "location" in node_info:
            return Location(node_info["location"]["x"], node_info["location"]["y"])
        return None
    
    def status(self, node_name=None, link_args=None):
        """Return the status of the scenario."""
        nodes = self.get_nodes()
        links = self.get_links()
        return nodes, links