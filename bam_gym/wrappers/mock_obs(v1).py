# PYTHON
import gymnasium as gym

# This works great, but doesn't work for vectorized envs!
# Which is honestly fine likely lol...

class MockObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # print(self)

    def observation(self, obs):
        return self.env.observation_space.sample()

