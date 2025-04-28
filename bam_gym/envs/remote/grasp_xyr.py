#!/usr/bin/env python3

# PYTHON

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2

# BAM
from bam_gym.envs import BamEnv

from bam_gym.transport import RoslibpyTransport, CustomTransport
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode




class GraspXYR(BamEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, transport, img_size=(640, 240), n_x=100, n_y=100, n_rz=16, n_obj_class=10, render_mode=None):
            
        super().__init__(transport, render_mode)
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
        self.action_space = spaces.MultiDiscrete([n_x, n_y, n_rz])

        self.observation_space = spaces.Dict({
            "class_ids": spaces.Sequence(
                spaces.Discrete(self.n_obj_class)
            ),
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
            "seg": spaces.Box(
                low=0,
                high=255,  # typically segmentation mask is uint8, 0-255
                shape=(self.img_height, self.img_width, 1),   # (H, W, 1)
                dtype=np.uint8
            ),
        })


        # For rendering the frame
        self.surface = None

    def reset(self, seed=None, options=None):
        self._reset(seed=seed)

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