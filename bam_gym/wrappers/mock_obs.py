# PYTHON
import gymnasium as gym

# This works great, but doesn't work for vectorized envs!
# Which is honestly fine likely lol...

# https://gymnasium.farama.org/_modules/gymnasium/core/#ObservationWrapper

# The pattern in Gym is to build up a chain of simple wrappers

class MockObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return self.env.observation_space.sample()

