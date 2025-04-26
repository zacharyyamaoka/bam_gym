from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from bam_gym_env.envs import BamEnv
from bam_gym_env.transport import RoslibpyTransport, CustomTransport
from bam_gym_env.ros_types.bam_srv import GymAPIRequest, GymAPIResponse, RequestType
from bam_gym_env.ros_types.bam_msgs import ErrorCode, GymAction
"""
Discretize grasping space

See: 'RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images' https://arxiv.org/pdf/2103.02184

See: https://github.com/GouMinghao/rgb_matters/blob/main/rgbd_graspnet/data/utils/convert.py#L78

See: https://github.com/GouMinghao/rgb_matters/blob/main/rgbd_graspnet/data/utils/generate_anchor_matrix.py#L24

"""

class GraspV1(BamEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, transport, render_mode=None):
            
        super().__init__(transport, render_mode)

        self.view_vectors = 120
        self.rotation_angles = 6
        self.img_shape = (1200,800,3)
        self.n_actions = self.img_shape[0] * self.img_shape[1] * self.view_vectors * self.rotation_angles

        self.action_space = spaces.Discrete(self.n_actions, start=1) # 0 is for empty action

        self.observation_space = spaces.Box(low=0, high=255, shape=self.img_shape, dtype=np.uint8)

        self.env_name = "grasp_v1"
        self.response = GymAPIResponse(dict())

    def reset(self, seed=None, options=None):
        
        response = self._reset(seed, options)

        self._render()

        return response.to_reset_tuple()

    def step(self, action):

        # convert from action in request
        request = GymAPIRequest()
        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name
        request.discrete_action = request.ensure_list(action)

        # convert from response into standard tuple
        response: GymAPIResponse = self._step(request)

        self._render()

        return response.to_step_tuple()
    
    def render(self):
        return self._render(self)

    def close(self):
        return self._close()
