"""Contextual thresholding for trust-based decisions."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union

class ContextualThreshold:
    """Contextual thresholding for trust-based decisions."""
    
    def __init__(self, config: Dict = None):
        """Initialize the contextual threshold.
        
        Args:
            config: Configuration for the threshold
                - base_threshold: Base trust threshold value
                - qos_weight: Weight for QoS requirements
                - congestion_weight: Weight for network congestion
                - attack_weight: Weight for detected attacks
                - criticality_weight: Weight for task criticality
                - min_threshold: Minimum threshold value
                - max_threshold: Maximum threshold value
        """
        self.config = config or {
            'base_threshold': 0.3,
            'qos_weight': 0.1,
            'congestion_weight': 0.05,
            'attack_weight': 0.2,
            'criticality_weight': 0.1,
            'min_threshold': 0.2,
            'max_threshold': 0.6
        }
        
        # Monitoring variables
        self.attack_level = 0.0  # Current attack level (0.0-1.0)
        self.congestion_level = 0.0  # Current congestion level (0.0-1.0)
        self.recent_thresholds = []  # Keep track of recent thresholds
    
    def compute_threshold(self, 
                         task: Optional[Dict] = None, 
                         network_state: Optional[Dict] = None) -> float:
        """Compute threshold based on context.
        
        Args:
            task: Task information (criticality, QoS requirements, etc.)
            network_state: Current network state (congestion, detected attacks, etc.)
            
        Returns:
            Computed threshold value
        """
        # Start with base threshold
        threshold = self.config['base_threshold']
        
        # Adjust for task criticality and QoS
        if task is not None:
            # Task criticality (higher -> stricter threshold)
            criticality = task.get('criticality', 0.5)
            threshold += criticality * self.config['criticality_weight']
            
            # QoS requirements (higher -> stricter threshold)
            qos_requirement = task.get('qos_requirement', 0.5)
            threshold += qos_requirement * self.config['qos_weight']
        
        # Adjust for network state
        if network_state is not None:
            # Congestion level (higher -> stricter threshold)
            self.congestion_level = network_state.get('congestion', self.congestion_level)
            threshold += self.congestion_level * self.config['congestion_weight']
            
            # Attack level (higher -> stricter threshold)
            attack_detection = network_state.get('attack_detected', self.attack_level)
            threshold += attack_detection * self.config['attack_weight']
        
        # Clamp to valid range
        threshold = max(self.config['min_threshold'], min(self.config['max_threshold'], threshold))
        
        # Keep track of recent thresholds
        self.recent_thresholds.append(threshold)
        if len(self.recent_thresholds) > 10:
            self.recent_thresholds.pop(0)
        
        return threshold
    
    def update_attack_level(self, attack_level: float):
        """Update the current attack level.
        
        Args:
            attack_level: New attack level (0.0-1.0)
        """
        self.attack_level = attack_level
    
    def update_congestion_level(self, congestion_level: float):
        """Update the current congestion level.
        
        Args:
            congestion_level: New congestion level (0.0-1.0)
        """
        self.congestion_level = congestion_level
    
    def get_average_threshold(self, window: int = 5) -> float:
        """Get the average threshold over a window of recent values.
        
        Args:
            window: Number of recent thresholds to average
            
        Returns:
            Average threshold
        """
        window = min(window, len(self.recent_thresholds))
        if window == 0:
            return self.config['base_threshold']
        
        return sum(self.recent_thresholds[-window:]) / window