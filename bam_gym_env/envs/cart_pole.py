from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from bam_gym_env.envs import BamEnv
from bam_gym_env.transport import RoslibpyTransport, CustomTransport, GymAPIRequest, GymAPIResponse, RequestType


"""
In here we wrap the GymAPI functionality and present a familar interface to other gym environments

"""

class CartPole(BamEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, transport, render_mode=None):
            
        super().__init__(transport, render_mode)

        # CartPole has 4 float observations
        self.observation_space = spaces.Box(
            low=np.array([-4.8, -np.inf, -0.418, -np.inf], dtype=np.float32),
            high=np.array([4.8, np.inf, 0.418, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)  # Left or right

        self.env_name = "cart_pole"
        self.response = GymAPIResponse(dict())

        # You can access the saved response via the parent class
    def reset(self, seed=None, options=None):
        
        response = self._reset(seed, options)

        self._render()

        return response.to_reset_tuple()

    def step(self, action):

        # convert from action in request
        request = GymAPIRequest()
        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name
        request.discrete_action = [action]

        # convert from response into standard tuple
        response: GymAPIResponse = self._step(request)

        self._render()

        return response.to_step_tuple()
    
    def render(self):
        return self._render(self)

    def close(self):
        return self._close()
