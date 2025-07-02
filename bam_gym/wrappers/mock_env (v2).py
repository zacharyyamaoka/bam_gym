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

- Intially was about mocking the transports, but then I added mock_transport=True
- Then was a standalone env
- Then I realised it was really a gym wrapper! 
- https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/


The reward_func and terminated_func are not useful when wrapping from the client factory...

however, in testing, they are very helpful, where I more explcictly set a reward...

Mabye better to just have a different wrapped though? idk not critical path.

"""

# would be cool if it worked for even CartPole standard lol...
class MockEnv(gym.Wrapper):
    # Verifies actions are in correct space, and returns mock observations in the correct space.
    # This supports setting any bam_api observatino space... see the GenericGymClient for more details.
    def __init__(
            self,
            env: gym.Env | gym.vector.VectorEnv,
            reward_func: Optional[Callable] = None,
            terminated_func: Optional[Callable] = None
            ):

        # override the transport with MockTransport()
        super().__init__(env)
        self.env: gym.Env | gym.vector.VectorEnv #assigned internally
        self.reward_func = reward_func
        self.terminated_func = terminated_func
        self.num_envs = getattr(self.env, "num_envs", 1)  # Handle both single and vector environments
        # self.env_name = "MockEnv"

    def reset(self, seed=None, options=None):
        """ Override reset in generic gym client"""


        observation = self.env.observation_space.sample()


        return (observation, {})
    
    def step(self, action):
        # to differentiate between list and np arrays, only lists should be used, otherwise you cannot tell if its a multi-dimensional action or multi-env action
        if isinstance(action, (list, tuple)):
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

        # if auto mask is enable you can just assign directly!
        # observations = self.observation_space.sample()
        
        # Generate data for each environment
        for env_idx in range(self.num_envs):
            # Sample observation from the observation space
            # obs = self.observation_space.sample()
            obs = self.env.single_observation_space.sample()
            observations.append(obs)

            # Use reward_fn if provided, otherwise random reward
            if self.reward_func is not None:
                reward = self.reward_func(step_count=self.step_count, env_idx=env_idx, obs=obs, action=action[env_idx])
            else:
                reward = np.random.uniform(-1.0, 1.0)
            rewards.append(reward)
            
            # Use terminated_fn if provided, otherwise always False
            if self.terminated_func is not None:
                is_terminated = self.terminated_func(step_count=self.step_count, env_idx=env_idx, obs=obs, action=action[env_idx])
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
    

