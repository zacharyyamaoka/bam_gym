from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2

from bam_gym.transport import RoslibpyTransport, MockTransport
from ros_py_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from ros_py_types.bam_msgs import ErrorCode

class GraspXY(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, 
                 render_mode=None, 
                 screen_size=(256, 256),
                 num_circles=5,
                 min_radius=5,
                 max_radius=20,
                 circle_color=None,
                 background_color=None):

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Env settings
        self.screen_width, self.screen_height = screen_size
        self.num_circles = num_circles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.circle_color = circle_color if circle_color else (255, 0, 0)  # Red
        self.background_color = background_color if background_color else (0, 0, 0)  # Black

        # Action = discrete (pixel x and y)
        self.action_space = spaces.MultiDiscrete([self.screen_width, self.screen_height])

        # Observation = RGB image
        self.observation_space = spaces.Box(0, 255, (self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Circles storage: each circle = (x, y, radius)
        self.circles = []

        # For rendering the frame
        self.surface = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize circles
        rng = np.random.default_rng(seed)
        self.circles = []
        for _ in range(self.num_circles):
            x = rng.integers(self.max_radius, self.screen_width - self.max_radius)
            y = rng.integers(self.max_radius, self.screen_height - self.max_radius)
            r = rng.integers(self.min_radius, self.max_radius)
            self.circles.append((x, y, r))

        # Create blank frame
        obs = self._draw_frame()
        info = {}

        return obs, info

    def step(self, action):
        x_guess, y_guess = action

        # Check if guess hits any circle
        reward = -1
        for cx, cy, cr in self.circles:
            if np.sqrt((x_guess - cx) ** 2 + (y_guess - cy) ** 2) <= cr:
                reward = 1
                break

        # Always single step episode
        terminated = True
        truncated = False
        info = {}

        # Update frame without guess mark
        obs = self._draw_frame(show_guess=None)

        if self.render_mode == "human":
            self._render_frame(self._draw_frame(show_guess=(x_guess, y_guess)))

        return obs, reward, terminated, truncated, info
        
    
    def _draw_frame(self, show_guess=None):
        frame = np.full((self.screen_height, self.screen_width, 3), self.background_color, dtype=np.uint8)

        for cx, cy, cr in self.circles:
            cv2.circle(frame, (cx, cy), cr, self.circle_color, -1)

        # Draw guess marker if provided
        if show_guess is not None:
            gx, gy = show_guess
            length = 5
            color = (255, 255, 255)  # White
            thickness = 1
            cv2.line(frame, (gx - length, gy), (gx + length, gy), color, thickness)
            cv2.line(frame, (gx, gy - length), (gx, gy + length), color, thickness)

        return frame
    
    def _render_frame(self, frame):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.window.blit(surface, (0, 0))
        pygame.display.update()
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        # Render latest frame
        if self.surface is not None:
            self._render_frame(self.surface)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()