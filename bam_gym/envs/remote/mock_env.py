

# PYTHON
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

# BAM
from bam_gym.envs.remote.bam_env import BamEnv
from bam_gym.transport import RoslibpyTransport, MockTransport
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode, GymAction, GymFeedback
from bam_gym.utils.utils import ensure_list

"""
Mock Env doesn't require any server to be running. Use this during testing

"""
class MockEnv(BamEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, transport, render_mode=None,  **kwargs):
            
        # override the transport with MockTransport()
        super().__init__(MockTransport(), render_mode)

        self.env_name = "MockEnv"

        self.action_space = spaces.Sequence(
            spaces.Discrete(2)  # Each action is still "Left" (0) or "Right" (1)
        )

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
        
    def step(self, action):

        response: GymAPI_Response = self._step(GymAPI_Request())

        (observations, rewards, terminated, truncated, infos) = response.to_step_tuple()

        return (observations, rewards, terminated, truncated, infos)
    

