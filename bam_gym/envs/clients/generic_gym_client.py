#!/usr/bin/env python3

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

""" Design Notes:

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


Special client that can appear as both a standard and psuedo-vectorized environment.

    See Vector Env Notes: https://gymnasium.farama.org/api/vector/
    Info on autoreset_mode: https://farama.org/Vector-Autoreset-Mode

    
The observations are always dicts or lists, so instead of doing a generic type decleration, we use dicts.

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore

"""

from typing import Any, Tuple, List, Dict, Optional

ResetReturn = Tuple[Dict, Dict[str, Any]] 
VecResetReturn = Tuple[List[Dict], Dict[str, Any]]

StepReturn = Tuple[Dict, float, bool, bool, Dict[str, Any]] 
VecStepReturn = Tuple[List[Dict], List[float], List[bool], List[bool], Dict[str, Any]]

class GenericGymClient(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "autoreset_mode": [gym.vector.AutoresetMode.SAME_STEP], "render_fps": 30}

    def __init__(self,
                 transport: RoslibpyTransport | MockTransport = MockTransport(),

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
                 autoreset_mode = gym.vector.AutoresetMode.SAME_STEP,
                 automask: bool = True,
                 ):
        #region - Assign to member variables
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        assert autoreset_mode is None or autoreset_mode in self.metadata["autoreset_mode"]
        self.autoreset_mode = autoreset_mode  # for vector environments

        # Store observation space parameters as member variables
        self.transport = transport
        self.num_envs = num_envs # for vector environments
        self.vec = self.num_envs > 1  # is this a vector environment?
        self.n_obs = n_obs
        self.n_color = n_color
        self.n_depth = n_depth
        self.n_detection = n_detection
        self.n_pose = n_pose
        self.automask = automask  # helper to automatically samply correct length of sequences spaces
        #endregion - Assign to member variables

        self._init_observation_space()  
        self._init_action_space() 
        self._init_render()  

        # Metrics
        self.step_count = 0
        self.reset_count = 0

        # Used by remote servers to identify the environment
        # Should get overriden by child class
        self.env_name = "bam_env" 
        
        # Commented out, beacuse going for stateless design
        # [Update] actually it would be nice to access from the outside
        self.request =  GymAPI_Request() 
        self.response = GymAPI_Response()

    def _init_observation_space(self):
        # GymAPI defines standard observation space, any env will be a subset of these:
        # If request succesful, then return an observation for each action
        # the dict can be empty, the the indexes should always line up!
        # For now I will not dot he same for Action Space as that is more custom for each environment
        obs_dict = spaces.Dict({})
        obs_mask = {} 
        # TODO this work for now but at somepoint will need to allow for nested masks in case there are sequences at a lower level
        # like with the segment 2d array

        if self.n_obs:
            obs_dict["obs_names"] = spaces.Sequence(spaces.Text(max_length=32))
            obs_dict["obs"] = spaces.Sequence(spaces.Box(low=-1, high=1, shape=(self.n_obs,), dtype=np.float32))
            obs_mask["obs_names"] = (self.n_obs, None)
            obs_mask["obs"] = (self.n_obs, None)
        if self.n_color:
            obs_dict["color_img"] = spaces.Sequence(custom_spaces.color_img_space())
            obs_mask["color_img"] = (self.n_color,)

        if self.n_depth:
            obs_dict["depth"] = spaces.Sequence(custom_spaces.depth_img_space())
            obs_mask["depth"] = (self.n_depth,)

        if self.n_color or self.n_depth:
            obs_dict["camera_info"] = spaces.Sequence(custom_spaces.camera_info_space())
            obs_mask["camera_info"] = (max(self.n_color, self.n_depth),)

        if self.n_pose:
            obs_dict["pose_names"] = spaces.Sequence(spaces.Text(max_length=32))
            obs_dict["pose"] = spaces.Sequence(custom_spaces.pose_stamped_space())
            obs_mask["pose_names"] = (self.n_pose, None)
            obs_mask["pose"] = (self.n_pose, None)
        # Add segments to observation space
        if self.n_detection:
            obs_dict["segments"] = spaces.Sequence(custom_spaces.segment2darray_space())
            obs_mask["segments"] = (self.n_detection, None)


        # Store single environment observation space
        self.obs_mask = obs_mask
        self.observation_space = obs_dict
        self.single_observation_space = None # always having this define does make life easier, but may lead to not precise behaviour

        if self.vec:
            self.single_observation_space = self.observation_space
            self.observation_space = spaces.Sequence(self.single_observation_space)

        if self.automask:
            if self.vec:
                self.single_observation_space = custom_spaces.MaskedSpaceWrapper(self.single_observation_space, mask=self.obs_mask) # type: ignore
                # add another level of masking on space.Sequence (n, mask_for_space)
                # https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Sequence.sample
                self.observation_space = custom_spaces.MaskedSpaceWrapper(self.observation_space, mask=(self.num_envs, self.obs_mask))
            else:
                self.observation_space = custom_spaces.MaskedSpaceWrapper(self.observation_space, mask=self.obs_mask)

    def _init_action_space(self):
        # Action space will be set by child classes
        # For vector environments, we'll need to create a batched version
        self.single_action_space = None 
        # To be set by child classes
         # Place holder for now, otherwise PassiveEnvChecker will complain
        self.action_space = spaces.Discrete(1) 

    def _init_render(self):
        # Params from pygame rendering
        # Env's can always implement their own custom game, 
        # but idea is to provide a basic GUI for viewing BamAPI
        # Use observation space flags from __init__ to config GUI
        self.window = None
        self.clock = None

    @property
    def success(self) -> bool:
        return self.response.header.error_code.value == ErrorType.SUCCESS
        # return self.transport.success

    def reset(self, seed=None, options=None) -> ResetReturn | VecResetReturn:
        """ Default reset(). Should work for most envs, but you can override if desired"""

        # Get GymAPI_Response from reset()
        response: GymAPI_Response = self._reset(seed, options)

        # Convert to (observation, info)
        (observations, infos) = response.to_reset_tuple(self.vec)

        self._render() # checks internally for render modes

        return (observations, infos) # type: ignore
    
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
        self.request = request

        self.response: GymAPI_Response = self.transport.step(request) # type: ignore
        return self.response
    
    def step(self, action) -> StepReturn | VecStepReturn: 
        """
        To be implemented in parent class.
        - Take a action of type (spaces) and fill a GymAPI_Request msg
        """
        self.request = GymAPI_Request()

        self.response: GymAPI_Response = self._step(self.request)

        (observations, rewards, terminated, truncated, infos) = self.response.to_step_tuple(self.vec)

        self._render()

        return (observations, rewards, terminated, truncated, infos) # type: ignore
    
    def _step(self, request: GymAPI_Request) -> GymAPI_Response:
        """ Helper function that should be called by step()"""
        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name
        self.request = request
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