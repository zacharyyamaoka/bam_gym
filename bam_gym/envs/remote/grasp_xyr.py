#!/usr/bin/env python3

# PYTHON

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2

# BAM
from bam_gym.envs.remote.bam_env import BamEnv

from bam_gym.transport import RoslibpyTransport, MockTransport
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode, GymAction

from bam_gym.utils.utils import ensure_list



class GraspXYR(BamEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, transport, img_size=(640, 240), n_x=100, n_y=100, n_rz=16, n_obj_class=10, render_mode=None):
            
        super().__init__(transport, render_mode)

        self.env_name = "GraspXYR" 

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Env settings
        self.img_height = img_size[0]
        self.img_width = img_size[1]
        self.n_x = n_x
        self.n_y = n_y
        self.n_rz = n_rz
        self.n_obj_class = n_obj_class

        # TODO how to pass in values for multiple arms...?

        self.action_space = spaces.Sequence(
            spaces.MultiDiscrete([n_x, n_y, n_rz])
        )

        self.observation_space = spaces.Dict({
            # Class Ids to Pick
            "target_ids": spaces.Sequence(spaces.Discrete(self.n_obj_class)),
            "obs": spaces.Box(
                low=np.array([-4.8, -np.inf, -0.418, -np.inf], dtype=np.float32),
                high=np.array([4.8, np.inf, 0.418, np.inf], dtype=np.float32),
                dtype=np.float32
            ),
            "color": spaces.Box(
                low=0,
                high=255,
                shape=(self.img_height, self.img_width, 3),   # (H, W, C)
                dtype=np.uint8
            ),
            "depth": spaces.Box(
                low=0.0,
                high=10.0,  # or whatever max depth makes sense (meters usually)
                shape=(self.img_height, self.img_width, 1),   # (H, W, 1) for single channel
                dtype=np.float32
            ),
            "seg_ids": spaces.Sequence(spaces.Discrete(self.n_obj_class)),

            # List of Polygons: list of (x, y) vertices
            "seg_mask": spaces.Sequence(
                spaces.Box(
                    low=np.array([0, 0], dtype=np.uint8),
                    high=np.array([self.img_width, self.img_height], dtype=np.uint8),
                    dtype=np.float32
                )   
            )
        })


        # For rendering the frame
        self.surface = None

    def reset(self, seed=None):
        # Get GymAPI_Response from reset()


        # The Env Server will read these values to configure the enviornment
        config_request = GymAPI_Request()
        msg = GymAction()
        # Put variables that change towards the ends so indexs stay more consistent
        msg.discrete_action = [self.img_height, self.img_width, self.n_obj_class, self.n_x, self.n_y, self.n_rz]
        config_request.action = [msg]

        response: GymAPI_Response = self._reset(seed, config_request)

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
        # action_msg.discrete_action = ensure_list(action)
        # request.action = [action_msg]

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
