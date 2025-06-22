#!/usr/bin/env python3

"""
Provide a template and common functionality for all Bam Environments

- For all common classes, there is a step() [external] and _step() [internal]
- The step, reset, close, etc. provide a good basic template for what the functions can look like
- The _step, _reset, _close, etc. provide basic functinoality for dealing with GymAPI - don't override these.
- To create a custom env, you just need to provide a thin wrapper that customizes the BamEnv as needed

Transports are injected into the env.
- This 'seperation of concerns' allows for different transports, roslibpy, rclpy, mock, etc. to be used

GUI should be injected into env.
- Right now its pygame, but It should be able to use pygame, or foxglove, etc. or mabye its simpler to just have pygame? 
- This is not critical can come later as needed

TODO:
- Support a vectorize flag, to switch between working with a single agent to many
"""

# BAM
from bam_gym.transport import RoslibpyTransport, MockTransport
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode, ErrorType
from bam_gym.envs import custom_spaces

# PYTHON
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2

from typing import Tuple, List, Dict

class BamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 transport: RoslibpyTransport | MockTransport,

                 # Observation Space
                 obs = False,
                 color = False,
                 depth = False,
                 detection = False,
                 pose = False,

                 # Misc
                 render_mode=None
                 ):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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
        observation_dict = spaces.Dict({})

        if obs:
            observation_dict["obs_names"] = spaces.Sequence(spaces.Text(max_length=32))
            observation_dict["obs"] = spaces.Sequence(spaces.Box(low=-1, high=1, shape=None, dtype=np.float32))
        if color:
            observation_dict["color_img"] = custom_spaces.color_img_space()
        if depth:
            observation_dict["depth"] = custom_spaces.depth_img_space()
        if color or depth:
            observation_dict["camera_info"] = custom_spaces.camera_info_space()
        if pose:
            observation_dict["pose_names"] = spaces.Sequence(spaces.Text(max_length=32))
            observation_dict["pose"] = spaces.Sequence(custom_spaces.pose_space())
        # Add segments to observation space
        if detection:
            observation_dict["segments"] = spaces.Sequence(custom_spaces.segment2darray_space())

        self.observation_space = spaces.Sequence(observation_dict)

    @property
    def success(self) -> bool:
        return self.response.header.error_code.value == ErrorType.SUCCESS
        # return self.transport.success

    def reset(self, seed=None) -> Tuple[List, Dict]:
        """ Default reset(). Should work for most envs, but you can override if desired"""

        # Get GymAPI_Response from reset()
        response: GymAPI_Response = self._reset(seed)

        # Convert to (observation, info)
        (observations, infos) = response.to_reset_tuple()

        self._render() # checks internally for render modes

        return (observations, infos)
    
    def _reset(self, seed=None, request: GymAPI_Request = None)-> GymAPI_Response:
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
        self.response: GymAPI_Response = self.transport.step(request)

        return self.response
    
    def step(self, action: spaces.Sequence) -> Tuple[List, List, List, List, Dict]:
        """
        To be implemented in parent class.
        - Take a action of type (spaces) and fill a GymAPI_Request msg
        """
        assert False, "Implement this is parent class"

    def _step(self, request: GymAPI_Request) -> GymAPI_Response:
        """ Helper function that should be called by step()"""
        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name
        self.response: GymAPI_Response = self.transport.step(request)
        return self.response

    def render(self):
        """ Default render() override if desired"""
        return self._render(self)
    
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
        self.response = self.transport.step(request)

        return self.response