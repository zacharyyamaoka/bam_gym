# PYTHON
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple

# BAM
from bam_gym.envs.clients.generic_gym_client import GenericGymClient
from bam_gym.transport import RoslibpyTransport, MockTransport
from ros_py_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from ros_py_types.bam_msgs import ErrorCode, GymAction, GymFeedback
from bam_gym.utils.utils import ensure_list

"""
Mock Env doesn't require any server to be running. Use this during testing

The purpose of this is to make a rapidly reconfigurable environment to help for testing...

It shouldn't impose any resitrictions on the actions...

"""
class MockEnv(GenericGymClient):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # This supports setting any bam_api observatino space... see the GenericGymClient for more details.
    def __init__(self,
    env_to_mock: Optional[gym.Env] = None,
    reward_fn: Optional[Callable] = None,
    terminated_fn: Optional[Callable] = None,
    **kwargs):
        # override the transport with MockTransport()
        super().__init__(MockTransport(), **kwargs)
        self.env_name = "MockEnv"

        if env_to_mock is not None:
            self.env_to_mock = env_to_mock
            self.action_space = env_to_mock.action_space
            self.observation_space = env_to_mock.observation_space
        else:
            self.env_to_mock = None
            self.action_space = spaces.Discrete(1)
            # self.observation_space = Will be taken from GenericGymClient

        self.single_action_space = self.action_space
        self.reward_fn = reward_fn
        self.terminated_fn = terminated_fn

    def reset(self, seed=None, options=None):
        """ Override reset in generic gym client"""

        (observations, rewards, terminated, truncated, infos) = self.step([None]*self.num_envs)
        self.step_count -= 1

        return (observations, infos)
        
    def step(self, action):
        # to differentiate between list and np arrays, only lists should be used, otherwise you cannot tell if its a multi-dimensional action or multi-env action
        if isinstance(action, list):
            n_actions = len(action)
        else:
            n_actions = 1
            action = [action]

        assert n_actions == self.num_envs, f"Number of actions {n_actions} must match number of environments {self.num_envs}"
        
        # Initialize lists for each return value
        observations = []
        rewards = []
        terminated = []
        truncated = []
        infos = {}
        
        # Generate data for each environment
        for env_idx in range(self.num_envs):
            # Sample observation from the observation space
            obs = self.observation_space.sample()
            observations.append(obs)
            
            # Use reward_fn if provided, otherwise random reward
            if self.reward_fn is not None:
                reward = self.reward_fn(step_count=self.step_count, env_idx=env_idx, obs=obs, action=action[env_idx])
            else:
                reward = np.random.uniform(-1.0, 1.0)
            rewards.append(reward)
            
            # Use terminated_fn if provided, otherwise always False
            if self.terminated_fn is not None:
                is_terminated = self.terminated_fn(step_count=self.step_count, env_idx=env_idx, obs=obs, action=action[env_idx])
            else:
                is_terminated = False
            terminated.append(is_terminated)
            truncated.append(False)
            
            # Add empty info dict for this environment
            infos[env_idx] = {}
        
        # If only one environment, unpack the lists
        if self.num_envs == 1:
            observations = observations[0]
            rewards = rewards[0]
            terminated = terminated[0]
            truncated = truncated[0]
            infos = infos[0]

        self.step_count += 1

        return (observations, rewards, terminated, truncated, infos)
    

