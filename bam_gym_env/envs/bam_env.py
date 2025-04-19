from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2

from bam_gym_env.transport import RoslibpyTransport, CustomTransport, GymAPIRequest, GymAPIResponse, RequestType

class BamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, transport: RoslibpyTransport | CustomTransport, render_mode=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # Go for stateless design
        # self.request =  GymAPIRequest()
        self.transport = transport

        self.response = GymAPIResponse(dict())

    def _reset(self, seed=None, options=None)-> GymAPIResponse:
        super().reset(seed=seed) # gym docs says to do this...

        request = GymAPIRequest()
        request.header.request_type = RequestType.RESET
        self.response: GymAPIResponse = self.transport.step(request)

        return self.response

    def _step(self, request: GymAPIRequest) -> GymAPIResponse:
        request.header.request_type = RequestType.STEP
        self.response = self.transport.step(request)
        return self.response

    def _render(self):
        
        if self.response.color_img is None:
            return 
        
        # Convert BGR to RGB for PyGame
        # rgb_image = cv2.cvtColor(self.response.color_img, cv2.COLOR_BGR2RGB)
        rgb_image = self.response.color_img

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

        request = GymAPIRequest()
        request.header.request_type = RequestType.CLOSE
        self.response = self.transport.step(request)

        return self.response