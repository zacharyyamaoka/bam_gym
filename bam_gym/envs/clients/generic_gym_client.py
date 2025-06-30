#!/usr/bin/env python3

"""
Provide a template and common functionality for all Bam Environments

- For all common classes, there is a step() [external] and _step() [internal]
- The step, reset, close, etc. provide a good basic template for what the functions can look like
- The _step, _reset, _close, etc. provide basic functinoality for dealing with GymAPI - don't override these.
- To create a custom env, you just need to provide a thin wrapper that customizes the GenericGymClient as needed

Transports are injected into the env.
- This 'seperation of concerns' allows for different transports, roslibpy, rclpy, mock, etc. to be used

GUI should be injected into env.
- Right now its pygame, but It should be able to use pygame, or foxglove, etc. or mabye its simpler to just have pygame? 
- This is not critical can come later as needed

Vector Environment Support:
- Supports being wrapped in vector environments while maintaining a single client in the background
- All vector operations go through the same GymAPI transport
- Provides required attributes: num_envs, single_observation_space, single_action_space
"""

# BAM
from bam_gym.transport import RoslibpyTransport, MockTransport
from ros_py_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from ros_py_types.bam_msgs import ErrorCode, ErrorType
from bam_gym.envs import custom_spaces

# PYTHON
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2

from typing import Tuple, List, Dict, Optional

class GenericGymClient(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "autoreset_mode": [True, False], "render_fps": 30}

    def __init__(self,
                 transport: RoslibpyTransport | MockTransport,

                 # Observation Space
                 n_obs: int = 0,
                 n_color: int = 0,
                 n_depth: int = 0,
                 n_detection: int = 0,
                 n_pose: int = 0,

                 # Vector Environment Support
                 num_envs: int = 1,

                 # Misc
                 render_mode: str | None = None,
                 autoreset_mode: bool = True
                 ):

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        # Vector environment support
        self.num_envs = num_envs
        self._is_vector_env = num_envs > 1
        self.autoreset_mode = autoreset_mode


        # Metrics
        self.step_count = 0
        self.reset_count = 0

        # Store observation space parameters as member variables
        self.n_obs = n_obs
        self.n_color = n_color
        self.n_depth = n_depth
        self.n_detection = n_detection
        self.n_pose = n_pose

        self.env_name = "bam_env" # this should get overriden by child class

        # Params from pygame rendering
        # Env's can always implement their own custom game, 
        # but idea is to provide a basic GUI for viewing BamAPI
        # Use observation space flags from __init__ to config GUI
        self.window = None
        self.clock = None

        # Go for stateless design
        # self.request =  GymAPI_Request()
        self.transport = transport
        self.response = GymAPI_Response()

        # GymAPI defines standard observation space, any env will be a subset of these:
        # If request succesful, then return an observation for each action
        # the dict can be empty, the the indexes should always line up!
        # For now I will not dot he same for Action Space as that is more custom for each environment
        obs_dict = spaces.Dict({})
        obs_mask = {} 
        # TODO this work for now but at somepoint will need to allow for nested masks in case there are sequences at a lower level
        # like with the segment 2d array

        if n_obs:
            obs_dict["obs_names"] = spaces.Sequence(spaces.Text(max_length=32))
            obs_dict["obs"] = spaces.Sequence(spaces.Box(low=-1, high=1, shape=(n_obs,), dtype=np.float32))
            obs_mask["obs_names"] = (n_obs, None)
            obs_mask["obs"] = (n_obs, None)
        if n_color:
            obs_dict["color_img"] = spaces.Sequence(custom_spaces.color_img_space())
            obs_mask["color_img"] = (n_color,)

        if n_depth:
            obs_dict["depth"] = spaces.Sequence(custom_spaces.depth_img_space())
            obs_mask["depth"] = (n_depth,)

        if n_color or n_depth:
            obs_dict["camera_info"] = spaces.Sequence(custom_spaces.camera_info_space())
            obs_mask["camera_info"] = (max(n_color, n_depth),)

        if n_pose:
            obs_dict["pose_names"] = spaces.Sequence(spaces.Text(max_length=32))
            obs_dict["pose"] = spaces.Sequence(custom_spaces.pose_space())
            obs_mask["pose_names"] = (n_pose, None)
            obs_mask["pose"] = (n_pose, None)
        # Add segments to observation space
        if n_detection:
            obs_dict["segments"] = spaces.Sequence(custom_spaces.segment2darray_space())
            obs_mask["segments"] = (n_detection, None)


        # Store single environment observation space
        self.obs_mask = obs_mask
        self.observation_space = obs_dict
        if self.num_envs > 1:
            self.single_observation_space = obs_dict
            self.observation_space = spaces.Sequence(self.observation_space)


        # Action space will be set by child classes
        # For vector environments, we'll need to create a batched version
        self.single_action_space = None  # To be set by child classes
        # self.action_space = None  # To be set by child classes

    @property
    def success(self) -> bool:
        return self.response.header.error_code.value == ErrorType.SUCCESS
        # return self.transport.success

    def reset(self, seed=None, options=None) -> Tuple[List, Dict]:
        """ Default reset(). Should work for most envs, but you can override if desired"""

        # Get GymAPI_Response from reset()
        response: GymAPI_Response = self._reset(seed, options)

        # Convert to (observation, info)
        (observations, infos) = response.to_reset_tuple()

        self._render() # checks internally for render modes

        return (observations, infos)
    
    def _reset(self, seed=None, options=None, request: GymAPI_Request | None = None)-> GymAPI_Response:
        """ Helper function that should be called by reset()"""
        super().reset(seed=seed) # gym docs says to do this...

        if request == None:
            request = GymAPI_Request()
            if seed != None:
                request.seed = seed
                # Ros msg seed = 0 is no seed
                # gym seed = None is no seed
        request.header.request_type = RequestType.RESET
        request.env_name = self.env_name
        response: GymAPI_Response = self.transport.step(request)
        self.response = response

        return self.response
    
    def step(self, action) -> Tuple[List, List, List, List, Dict]:
        """
        To be implemented in parent class.
        - Take a action of type (spaces) and fill a GymAPI_Request msg
        """
        assert False, "Implement this is parent class"

    def _step(self, request: GymAPI_Request) -> GymAPI_Response:
        """ Helper function that should be called by step()"""
        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name
        response: GymAPI_Response = self.transport.step(request)
        self.response = response
        self.step_count += 1

        return self.response

    def render(self):
        """ Default render() override if desired"""
        return self._render()
    
    def _render(self):
        """"""
        if self.render_mode == None:
            # print("Render mode None")
            return

        if len(self.response.feedback) == 0:
            print("[_render()] Cannot render as len(feedback) == 0")
            return 
               
        r = self.response.feedback[0]
        # Handle color_img as a list of images
        if len(r.color_img) == 0:
            print("No color images received")
            return 

        # Use the first image in the list for rendering
        # it should be decompressed already by the transport
        img0 = r.color_img[0]
        rgb_image = img0.data
        if not isinstance(rgb_image, np.ndarray):
            print(f"Cannot render, not valid image data type {type(rgb_image)}")
            return
        
        if self.render_mode == "human":
            # print("Render mode Human")
            height, width, _ = rgb_image.shape

            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((width, height))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))  # PyGame expects (width, height)
            self.window.blit(surface, (0, 0))
            pygame.display.update()
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return rgb_image
    
    def close(self) -> GymAPI_Response:
        """ Default close() override if desired"""

        return self._close()
    
    def _close(self) -> GymAPI_Response:

        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None  

        request = GymAPI_Request()
        request.header.request_type = RequestType.CLOSE
        request.env_name = self.env_name
        response = self.transport.step(request)
        self.response = response

        return self.response