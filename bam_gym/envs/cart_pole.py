from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from bam_gym_env.envs import BamEnv
from bam_gym_env.transport import RoslibpyTransport, CustomTransport
from bam_gym_env.ros_types.bam_srv import GymAPIRequest, GymAPIResponse, RequestType
from bam_gym_env.ros_types.bam_msgs import ErrorCode, GymAction, GymFeedback
from bam_gym_env.ros_types.utils import ensure_list

"""
In here we wrap the GymAPI functionality and present a familar interface to other gym environments

Go from normal gym to Ros Transport

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

        # You can access the saved response via the parent class
        # Don't save the response, stateless design
        
    def reset(self, seed=None, options=None):
        
        response = self._reset(seed, options)
        feedback = response.feedback[0]
        self._render()

        return feedback.to_reset_tuple()

    def step(self, action):

        # convert from action in request
        request = GymAPIRequest()

        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name

        action_msg = GymAction()
        action_msg.discrete_action = ensure_list(action)
        request.action = [action_msg]

        # convert from response into standard tuple
        response: GymAPIResponse = self._step(request)
        feedback: GymFeedback = response.feedback[0]

        self._render()

        return feedback.to_step_tuple()
    
    def render(self):
        return self._render(self)

    def close(self):
        return self._close()
