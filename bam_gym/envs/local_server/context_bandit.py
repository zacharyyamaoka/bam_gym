import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class ContextBandit(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, n_arms=5, n_state=10, render_mode=None, seed=None):
        super().__init__()

        self.n_arms = n_arms
        self.n_state = n_state
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Discrete(self.n_state)  # Always dummy 0

        self.render_mode = render_mode

        self.seed = seed
        self.rng = np.random.default_rng(seed)


        self.arm_thresholds = {}
        for state in range(self.n_state):
            exponents = self.rng.pareto(a=2.0, size=self.n_arms)
            exponents = np.clip(exponents, 0, 5)
            thresholds = exponents / exponents.max()
            self.arm_thresholds[state] = thresholds

        self.current_state = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_state = self.rng.integers(0, self.n_state)

        info = {}
        return self.current_state, info

    def step(self, action):
        threshold = self.arm_thresholds[self.current_state][int(action)]
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
        
        self.reset() # autorest

        return self.current_state, reward, terminated, truncated, info

    def render(self, sample_states=5):
        """Render bar charts of arm thresholds for the first X states."""

        # Another cool way to viz this would be drawing an image with pixel intesnity equal to threshold
        if self.arm_thresholds is None:
            print("Call reset() first before rendering.")
            return

        # Take the first `sample_states` states
        states = list(self.arm_thresholds.keys())[:sample_states]

        fig, axes = plt.subplots(len(states), 1, figsize=(10, 5), sharex=True)

        if len(states) == 1:
            axes = [axes]  # Ensure axes is iterable

        x = np.arange(self.n_arms)

        for ax, state in zip(axes, states):
            thresholds = self.arm_thresholds[state]
            ax.bar(x, thresholds, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Threshold')
            ax.set_title(f'State {state} Thresholds')
            ax.grid(True)

        axes[-1].set_xlabel('Arm')

        plt.tight_layout()
        plt.show()

    def close(self):
        pass
