#!/usr/bin/env python3

import gymnasium as gym
import bam_gym # you need to import to register
from bam_gym.utils import SampleSaver

env = gym.make("bam/ClassicBandit", n_arms=10, seed=1)


saver = SampleSaver("/home/bam/other_bam_packages/bam_gym/dataset",
                 "ClassicBandit",
                 "GPU"
                 )

observation, info = env.reset()
# env.render()

for _ in range(1000):
    # action = policy(observation)
    action = env.action_space.sample()

    next_observation, reward, terminated, truncated, next_info = env.step(action)
    
    print(f"\nStep Result:")
    print(f"Observation: {observation}")
    # print(f"Observation: (shape={observation.shape}, dtype={observation.dtype})")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    saver.save_sample(observation, action, reward, terminated, truncated, info)

    observation = next_observation
    info = next_info

    # if (terminated or truncated):
    #     observation, info = env.reset()

env.close()
saver.close()
