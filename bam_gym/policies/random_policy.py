#!/usr/bin/env python3

"""
Random policy implementations for BAM agents.
"""

from bam_gym.policies.generic_policy import GenericPolicy
from typing import Any, Dict, Tuple
import gymnasium as gym
class RandomPolicy(GenericPolicy):
    """
    Simple random policy that samples actions from the environment's action space.
    
    This is useful for baseline testing and exploration.
    """
    def __init__(self, ):
        super().__init__()
        
        self.last_observation: Any = None
        self.step_count: int = 0

    def _validate_environment(self, env: gym.Env) -> bool:
        """
        Validate that the environment has a sampleable action space.
        
        Returns:
            bool: True if environment is valid for random policy
        """
        try:
            
            # Try to sample once to ensure it works
            test_action = self.action_space.sample()
            
            return True
            
        except Exception as e:
            print(f"RandomPolicy environment validation failed: {e}")
            return False
    
    def step(self, observation, reward=None, terminated=None, truncated=None, info=None) -> Tuple[Any, Dict[str, Any]]:
        assert self.action_space is not None, "env_init() must be called before calling step"
        return self.action_space.sample(), {}
