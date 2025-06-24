import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2

class GraspBandit(gym.Env):

    def __init__(self,
                 n_arms=5,
                 image_size=(1, 1, 1),
                 seed=None):
        

        # Bandit settings
        self.n_arms = n_arms
        self.stateful = stateful
        self.screen_width, self.screen_height = screen_size
        self.n_pixels = self.screen_width * self.screen_height

        # Action space = select an arm
        self.action_space = spaces.Discrete(self.n_arms)

        # Observation space = 0 or 1 pixels (dtype int8)
        self.observation_space = spaces.Discrete(0)

        # RNG
        self.rng = np.random.default_rng(seed)

        # Bandit mappings
        self.bandit_means = {}
        self.bandit_stds = {}

        # Generate fixed bandits for known states if stateful
        if self.stateful:
            self.state_table = {}
            self._generate_bandits_stateful()
        else:
            self.bandit_means = self.rng.uniform(-1, 1, size=self.n_arms)
            self.bandit_stds = self.rng.uniform(0.05, 0.2, size=self.n_arms)

        # Track current state
        self.frame = None
        self.current_key = None

    def _generate_bandits_stateful(self):
        """Generate bandits for all possible states."""
        n_possible_states = 2 ** self.n_pixels
        for idx in range(n_possible_states):
            means = self.rng.uniform(-1, 1, size=self.n_arms)
            stds = self.rng.uniform(0.05, 0.2, size=self.n_arms)
            self.bandit_means[idx] = means
            self.bandit_stds[idx] = stds

    def _frame_to_key(self, frame):
        """Flatten frame and encode as integer key."""
        bits = frame.flatten()
        key = 0
        for bit in bits:
            key = (key << 1) | int(bit)
        return key

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random binary pixel map
        bits = self.rng.integers(0, 2, size=(self.screen_height, self.screen_width, 1), dtype=np.int8)
        self.frame = bits

        # Lookup key for stateful behavior
        if self.stateful:
            self.current_key = self._frame_to_key(self.frame)
        else:
            self.current_key = None

        info = {}
        return self.frame.copy(), info

    def step(self, action):
        if self.stateful:
            means = self.bandit_means[self.current_key]
            stds = self.bandit_stds[self.current_key]
        else:
            means = self.bandit_means
            stds = self.bandit_stds

        mean = means[action]
        std = stds[action]

        # Sample reward
        reward = self.rng.normal(mean, std)

        terminated = True
        truncated = False
        info = {
            "true_mean": mean,
            "true_std": std,
            "state_key": self.current_key
        }

        if self.render_mode == "human":
            self._render_frame()

        return self.frame.copy(), reward, terminated, truncated, info

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width * 50, self.screen_height * 50))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Upscale frame for visualization
        frame_vis = (self.frame.squeeze(-1) * 255).astype(np.uint8)
        frame_large = cv2.resize(frame_vis, (self.screen_width * 50, self.screen_height * 50), interpolation=cv2.INTER_NEAREST)
        frame_large = np.stack([frame_large] * 3, axis=-1)  # Grayscale to RGB

        surface = pygame.surfarray.make_surface(frame_large.swapaxes(0, 1))
        self.window.blit(surface, (0, 0))
        pygame.display.update()
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        if self.frame is not None:
            self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
