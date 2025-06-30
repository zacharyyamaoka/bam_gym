#!/usr/bin/env python3

"""
Custom Vector Environment for BAM Gym Environments

This vector environment wrapper is designed to work with GenericGymClient
while maintaining a single client in the background. All vector operations
go through the same GymAPI transport, but the wrapper handles batching
and unbatching of observations, actions, rewards, etc.

This is different from standard Gymnasium vector environments because:
1. Only one actual client/transport is used
2. All environments share the same underlying GymAPI connection
3. The wrapper handles the vectorization logic
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from .clients.generic_gym_client import GenericGymClient


class BamVectorEnv(gym.vector.VectorEnv):
    """
    Vector environment wrapper for BAM Gym environments.
    
    This wrapper maintains a single GenericGymClient in the background
    and handles vectorization by batching/unbatching operations.
    """
    
    def __init__(self, 
                 env_factory,
                 num_envs: int = 4,
                 **env_kwargs):
        """
        Initialize the vector environment.
        
        Args:
            env_factory: Function that creates a single environment instance
            num_envs: Number of parallel environments
            **env_kwargs: Additional arguments passed to env_factory
        """
        self.num_envs = num_envs
        
        # Create a single client that will handle all environments
        self.single_env = env_factory(**env_kwargs)
        
        # Set up the single environment for vector mode
        self.single_env.num_envs = num_envs
        self.single_env._is_vector_env = True
        
        # Get spaces from the single environment
        self.single_observation_space = self.single_env.single_observation_space
        self.single_action_space = self.single_env.single_action_space
        
        # Create batched spaces
        self.observation_space = self._create_batched_observation_space()
        self.action_space = self._create_batched_action_space()
        
        # Initialize state tracking
        self._observations = None
        self._rewards = None
        self._terminations = None
        self._truncations = None
        self._infos = None
        
        # Metadata
        self.metadata = self.single_env.metadata.copy()
        self.metadata["autoreset_mode"] = "next_step"
        
        # Render mode
        self.render_mode = self.single_env.render_mode
        
        # Environment spec
        self.spec = self.single_env.spec
        
        # Closed flag
        self.closed = False
    
    def _create_batched_observation_space(self):
        """Create a batched observation space based on the single environment's space."""
        single_space = self.single_observation_space
        
        if isinstance(single_space, spaces.Sequence):
            # For sequence spaces, we create a sequence of sequences
            return spaces.Sequence(single_space)
        elif isinstance(single_space, spaces.Dict):
            # For dict spaces, we need to batch each component
            batched_dict = {}
            for key, space in single_space.spaces.items():
                if isinstance(space, spaces.Box):
                    # For Box spaces, add batch dimension
                    low = np.tile(space.low, (self.num_envs, 1))
                    high = np.tile(space.high, (self.num_envs, 1))
                    batched_dict[key] = spaces.Box(low=low, high=high, dtype=space.dtype)
                else:
                    # For other spaces, use sequence
                    batched_dict[key] = spaces.Sequence(space)
            return spaces.Dict(batched_dict)
        else:
            # For other space types, use sequence
            return spaces.Sequence(single_space)
    
    def _create_batched_action_space(self):
        """Create a batched action space based on the single environment's space."""
        single_space = self.single_action_space
        
        if isinstance(single_space, spaces.Discrete):
            return spaces.MultiDiscrete([single_space.n] * self.num_envs)
        elif isinstance(single_space, spaces.Box):
            # For Box spaces, add batch dimension
            low = np.tile(single_space.low, (self.num_envs, 1))
            high = np.tile(single_space.high, (self.num_envs, 1))
            return spaces.Box(low=low, high=high, dtype=single_space.dtype)
        elif isinstance(single_space, spaces.MultiDiscrete):
            # For MultiDiscrete, repeat the nvec
            nvec = np.tile(single_space.nvec, (self.num_envs, 1))
            return spaces.MultiDiscrete(nvec)
        else:
            # For other space types, use sequence
            return spaces.Sequence(single_space)
    
    def reset(self, seed=None, options=None):
        """Reset all environments and return batched observations."""
        if seed is not None:
            # Set seed for the single environment
            self.single_env.reset(seed=seed, options=options)
        
        # Get observations from single environment
        observations, infos = self.single_env.reset(seed=seed, options=options)
        
        # Ensure observations are in the correct format
        if not isinstance(observations, list):
            observations = [observations] * self.num_envs
        
        # Store state
        self._observations = observations
        self._infos = infos
        
        return observations, infos
    
    def step(self, actions):
        """Take actions for all environments and return batched results."""
        # Convert batched actions to the format expected by the single environment
        # This depends on how the single environment expects to receive actions
        
        # For now, we'll assume the single environment can handle batched actions
        # In practice, you might need to modify the single environment's step method
        # to handle vectorized actions properly
        
        observations, rewards, terminations, truncations, infos = self.single_env.step(actions)
        
        # Ensure all returns are in the correct format
        if not isinstance(observations, list):
            observations = [observations] * self.num_envs
        if not isinstance(rewards, (list, np.ndarray)):
            rewards = [rewards] * self.num_envs
        if not isinstance(terminations, (list, np.ndarray)):
            terminations = [terminations] * self.num_envs
        if not isinstance(truncations, (list, np.ndarray)):
            truncations = [truncations] * self.num_envs
        
        # Convert to numpy arrays for consistency with Gymnasium vector envs
        rewards = np.array(rewards, dtype=np.float32)
        terminations = np.array(terminations, dtype=bool)
        truncations = np.array(truncations, dtype=bool)
        
        # Store state
        self._observations = observations
        self._rewards = rewards
        self._terminations = terminations
        self._truncations = truncations
        self._infos = infos
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        """Render the environment."""
        return self.single_env.render()
    
    def close(self):
        """Close the environment."""
        if not self.closed:
            self.single_env.close()
            self.closed = True
    
    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.single_env.unwrapped
    
    @property
    def np_random(self):
        """Return the environment's random number generator."""
        return self.single_env.np_random
    
    @property
    def np_random_seed(self):
        """Return the environment's random seed."""
        return self.single_env.np_random_seed


def make_bam_vec_env(env_factory, num_envs: int = 4, **env_kwargs):
    """
    Create a vectorized BAM environment.
    
    Args:
        env_factory: Function that creates a single environment instance
        num_envs: Number of parallel environments
        **env_kwargs: Additional arguments passed to env_factory
    
    Returns:
        BamVectorEnv: Vectorized environment
    """
    return BamVectorEnv(env_factory, num_envs, **env_kwargs) 