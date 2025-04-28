

# PYTHON
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

# BAM
from bam_gym.envs.remote.bam_env import BamEnv
from bam_gym.transport import RoslibpyTransport, CustomTransport
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode, GymAction, GymFeedback
from bam_gym.ros_types.utils import ensure_list

"""
In here we wrap the GymAPI functionality and present a familar interface to other gym environments

Go from normal gym to Ros Transport

"""

class CartPole(BamEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, transport, render_mode=None):
            
        super().__init__(transport, render_mode)

        self.env_name = "cart_pole"

        self.action_space = spaces.Sequence(
            spaces.Discrete(2)  # Each action is still "Left" (0) or "Right" (1)
        )

        # If request succesful, then return an observation for each action
        # the dict can be empty, the the indexes should always line up!
        self.observation_space = spaces.Sequence(
            spaces.Dict({
                "obs": spaces.Box(
                    low=np.array([-4.8, -np.inf, -0.418, -np.inf], dtype=np.float32),
                    high=np.array([4.8, np.inf, 0.418, np.inf], dtype=np.float32),
                    dtype=np.float32
                ),
                "color": spaces.Box(
                    low=0,
                    high=255,
                    shape=(480, 600, 3),  # (H, W, C)
                    dtype=np.uint8
                ),
            })
        )
        
    def reset(self, seed=None):

        # Get GymAPI_Response from reset()
        response: GymAPI_Response = self._reset(seed)

        # Convert to (observation, info)
        reset_tuple = response.to_reset_tuple()

        self._render() # checks internally for render modes

        return reset_tuple
    
    def step(self, action):

        # Convert from action to GymAPI_Request
        request = GymAPI_Request()

        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name

        action_msg = GymAction()
        action_msg.discrete_action = ensure_list(action)
        request.action = [action_msg]

        # Get response
        response: GymAPI_Response = self._step(request)

        # Convert from GymAPI_Response to (observation, reward, terminated, truncated, info)
        step_tuple = response.to_step_tuple()

        self._render()

        return step_tuple
    
    def render(self):
        return self._render(self)

    def close(self):
        return self._close()
