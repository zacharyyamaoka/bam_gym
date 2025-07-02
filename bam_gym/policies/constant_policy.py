#!/usr/bin/env python3

"""
Constant policy implementation for BAM agents.
"""

from typing import Any, Dict, Tuple
import numpy as np
from bam_gym.policies.generic_policy import GenericPolicy
import gymnasium as gym

class ConstantPolicy(GenericPolicy):
    """
    Policy that always returns the same action.
    
    Useful for testing and debugging.
    """
    
    def __init__(self, action=None, **kwargs):
        super().__init__(**kwargs)
        self.action = action
    
    def _validate_environment(self, env: gym.Env) -> bool:
        """
        Validate that the environment is compatible with constant policy.
        
        Returns:
            bool: True if environment is valid for constant policy
        """
        try:
            if self.action_space is None:
                print("ConstantPolicy requires a valid action_space")
                return False
            
            # If action is already set, validate it against the action space
            if self.action is not None:
                if hasattr(self.action_space, 'contains'):
                    if not self.action_space.contains(self.action):
                        print(f"ConstantPolicy action {self.action} is not in action space {self.action_space}")
                        return False
            
            return True
            
        except Exception as e:
            print(f"ConstantPolicy environment validation failed: {e}")
            return False
    
    def step(self, observation, reward=None, terminated=None, truncated=None, info=None) -> Tuple[Any, Dict[str, Any]]:
        """Return the constant action"""
        if self.action is None:
            if self.action_space is None:
                raise ValueError("Policy not initialized with environment. Call env_init() first.")
            # Default to first action if none specified
            if hasattr(self.action_space, 'n'):
                self.action = 0  # Discrete action space
            elif hasattr(self.action_space, 'shape') and self.action_space.shape is not None:
                self.action = np.zeros(self.action_space.shape)  # Continuous action space
            else:
                raise ValueError(f"Cannot determine default action for action space: {self.action_space}")
        
        return self.action, {}  