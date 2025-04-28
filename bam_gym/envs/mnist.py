import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from torchvision import datasets, transforms

from pathlib import Path

class MNISTEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.window_size = 256
        self.image_size = 28

        self.action_space = spaces.Discrete(10)  # digits 0–9
        self.observation_space = spaces.Box(0, 1, shape=(self.image_size, self.image_size), dtype=np.float32)

        # PyGame setup
        self.window = None
        self.clock = None

        # MNIST dataset
        data_dir = Path(__file__).resolve().parent.parent / "dataset"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = datasets.MNIST(root=str(data_dir), train=True, download=True,
                                      transform=transforms.ToTensor())
        self.current_index = None
        self.current_img = None
        self.current_label = None

        self._seed = None

    def _reset(self, seed=None):
        self.current_index = self.np_random.integers(0, len(self.dataset))
        self.current_img, self.current_label = self.dataset[self.current_index]
        self.current_img = self.current_img.squeeze(0).numpy()  # shape (28, 28)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset(seed)
        self._seed = seed

        return self.current_img

    def step(self, action):
        self.curr_action = action 

        # Render the current frame and action together
        # Makes more sense for supervised learning
        if self.render_mode == "human":
            self._render_frame()

        terminated = True
        reward = 1 if action == self.current_label else 0
        obs = self.current_img

        self._reset(self._seed)


        return obs, reward, terminated, False, {"label": self.current_label, "auto_reset":True}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        status_bar_height = 40  # Height for the bottom bar
        total_height = self.window_size + status_bar_height

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, total_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Full canvas including image + status bar
        canvas = pygame.Surface((self.window_size, total_height))
        canvas.fill((255, 255, 255))

        # Resize MNIST image to display size
        img = (self.current_img.T * 255).astype(np.uint8)  # Transpose to fix orientation
        surface = pygame.surfarray.make_surface(np.stack([img]*3, axis=-1))
        surface = pygame.transform.scale(surface, (self.window_size, self.window_size))
        canvas.blit(surface, (0, 0))

        # Draw bottom status bar
        if not hasattr(self, "curr_action"):
            self.curr_action = "N/A"
            
        if self.curr_action == self.current_label:
            bar_color = (100, 255, 100)  # green
        else:
            bar_color = (255, 100, 100)  # red

        pygame.draw.rect(canvas, bar_color, (0, self.window_size, self.window_size, status_bar_height))

        # Render text
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Action: {self.curr_action}  Label: {self.current_label}", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.window_size // 2, self.window_size + status_bar_height // 2))
        canvas.blit(text, text_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
