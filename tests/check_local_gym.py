#!/usr/bin/env python3

import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
from bam_gym.utils import print_step_result


observation, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()

    new_observation, reward, terminated, truncated, info = env.step(action)
    
    print_step_result(_, observation, action, reward, terminated, truncated, info)
    observation = new_observation

    if (terminated or truncated):
        observation, info = env.reset()
env.close()