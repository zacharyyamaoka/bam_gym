from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2

from bam_gym.transport import RoslibpyTransport, CustomTransport
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode, ErrorType

"""
Provide a template and common functionality for all Bam Environments

Use in child class by calling _step, _reset, _close, etc.

Seperation of concerns with transport allows for roslibpy or rclpy to be used

"""
class BamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, transport: RoslibpyTransport | CustomTransport, render_mode=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.env_name = "bam_env" # this should get overriden by child class

        # Params from pygame rendering
        # Env's can always implement their own custom game, 
        # but idea is to provide a basic GUI for viewing images
        self.window = None
        self.clock = None

        # Go for stateless design
        # self.request =  GymAPI_Request()
        self.transport = transport
        self.response = GymAPI_Response()

    @property
    def success(self) -> bool:
        return self.response.header.error_code.value == ErrorType.SUCCESS
        # return self.transport.success

    def _reset(self, seed=None, request: GymAPI_Request = None)-> GymAPI_Response:
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

    def _step(self, request: GymAPI_Request) -> GymAPI_Response:
        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name
        self.response: GymAPI_Response = self.transport.step(request)
        return self.response

    def _render(self):
        """"""
        if self.render_mode == None:
            return

        if len(self.response.feedback) == 0:
            print("[_render()] Cannot render as len(feedback) == 0")
            return 
               
        r = self.response.feedback[0]
        if r.color_img is None:
            print("No color image recivied")
            return 
        
        # Convert BGR to RGB for PyGame
        # rgb_image = cv2.cvtColor(r.color_img, cv2.COLOR_BGR2RGB)
        rgb_image = r.color_img.data
        if not hasattr(rgb_image, "shape"):
            print("Cannot render, empty color image")
            return
        
        if self.render_mode == "human":

            # Flip vertically if needed (OpenCV and PyGame may have different origins)
            # rgb_image = np.flipud(rgb_image)
            
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
    
    def _close(self):

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