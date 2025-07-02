#!/usr/bin/env python3

"""
Generic policy base class for BAM agents.

Policies define how agents select actions based on observations.
"""

import gymnasium as gym
from typing import Any, Dict, Optional, Tuple

class GenericPolicy():
    """
    Base class for all policies.
    
    A policy defines how an agent selects actions based on observations.
    Policies are callable objects that take observations and return actions.
    """
    
    def __init__(self, **kwargs):
        self.last_observation: Any = None
        self.step_count: int = 0
        self.ready: bool = True  # Policy is ready by default

    def env_init(self, env: gym.Env) -> bool:
        """
        Initialize the policy with environment information.
        
        Args:
            env: The gym environment to validate and use
            
        Returns:
            bool: True if environment is valid, False otherwise


        Design Notes:

        - This could be done in the init, but I think it makes sense to do it afterwards
        - Gives flexibility on the order of policy and environment initialization
        """
        try:
            # Basic environment validation
            if not isinstance(env, gym.Env):
                raise ValueError(f"Environment must be a gym.Env, got {type(env)}")
            
            if not hasattr(env, 'action_space') or env.action_space is None:
                raise ValueError("Environment must have a valid action_space")
                
            if not hasattr(env, 'observation_space') or env.observation_space is None:
                raise ValueError("Environment must have a valid observation_space")
            
            # Store environment information
            self.env: gym.Env = env
            self.action_space: gym.Space = env.action_space
            self.observation_space: gym.Space = env.observation_space
            
            # Call policy-specific validation
            self._validate_environment(env)
                
            return True
            
        except Exception as e:
            print(f"Environment validation failed: {e}")
            return False
    
    def _validate_environment(self, env: gym.Env) -> None:
        """
        Policy-specific environment validation.
        Override this method in subclasses to add specific validation.
        
        Raises:
            ValueError: If the environment is not valid for this policy

        I considered also returning a bool, but raising an error is more inline with intent,
        as by default you don't want to continue execution! You can always wrap in a try/except, to get the best of both worlds
        """
        pass

    def __call__(self, observation, reward=None, terminated=None, truncated=None, info=None) -> Tuple[Any, Dict[str, Any]]:
        """
        Select an action based on the current observation.
        
        Args:
            observation: The current observation from the environment
            
        Returns:
            action: The selected action
        """


        # Pre-step processing
        self.last_observation = observation
        self.step_count += 1
        
        action, action_info = self.step(observation, reward, terminated, truncated, info)

        # Post-step processing
        return action, action_info
        
    def step(self, observation, reward=None, terminated=None, truncated=None, info=None) -> Tuple[Any, Dict[str, Any]]:
        """
        Select an action based on the current observation.
        Override this method in subclasses.
        
        Args:
            observation: The current observation from the environment
            
        Returns:
            action: The selected action
        """
        return None, {}