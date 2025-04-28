import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class ClassicBandit(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, n_arms=5, render_mode=None, seed=None):
        super().__init__()

        self.n_arms = n_arms
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Discrete(1)  # Always dummy 0

        self.render_mode = render_mode

        self.seed = seed
        self.rng = np.random.default_rng(seed)


        exponents = self.rng.pareto(a=2.0, size=n_arms)
        exponents = np.clip(exponents, 0, 5)
        thresholds = exponents / exponents.max()
        self.arm_thresholds = thresholds


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        info = {}
        return 0, info

    def step(self, action):
        threshold = self.arm_thresholds[action]
        random_value = self.rng.uniform(0, 1)

        if random_value <= threshold:
            reward = 1
        else:
            reward = -1

        terminated = True
        truncated = False
        info = {
            "threshold": threshold,
            "random_value": random_value,
        }

        return 0, reward, terminated, truncated, info

    def render(self):
        """Render bar chart of arm thresholds."""
        if self.arm_thresholds is None:
            print("Call reset() first before rendering.")
            return

        x = np.arange(self.n_arms)
        thresholds = self.arm_thresholds

        plt.figure(figsize=(10, 6))
        plt.bar(x, thresholds, alpha=0.7)
        plt.axhline(0.5, color='black', linestyle='--', label="Random Guess Line (0.5)")
        plt.ylim(0, 1)

        plt.xlabel('Arm')
        plt.ylabel('Threshold')
        plt.title('Classic Bandit Arms: Thresholds (Power Law)')

        plt.grid(True)
        plt.show()

    def close(self):
        pass
