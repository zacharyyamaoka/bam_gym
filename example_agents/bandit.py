#!/usr/bin/env python3

import gymnasium as gym
import bam_gym # you need to import to register
env = gym.make("bam/ClassicBandit", n_arms=100)


observation, info = env.reset(seed=1)
env.render()

# for _ in range(100):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
    
#     print(f"\nStep Result:")
#     print(f"Action: {action}")
#     print(f"Observation: (shape={observation.shape}, dtype={observation.dtype})")
#     # print(f"Observation: {observation}")
#     print(f"Reward: {reward}")
#     print(f"Terminated: {terminated}")
#     print(f"Truncated: {truncated}")

#     if (terminated or truncated):
#         observation, info = env.reset()
# env.close()

