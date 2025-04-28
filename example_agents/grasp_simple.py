#!/usr/bin/env python3

import gymnasium as gym
# env = gym.make("CartPole-v1", render_mode="human")
import bam_gym # you need to import to register
# env = gym.make("bam/GridWorld-v0", render_mode="human")
env = gym.make("bam/GraspXY", num_circles=1, render_mode="human")


observation, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep Result:")
    print(f"Action: {action}")
    print(f"Observation: (shape={observation.shape}, dtype={observation.dtype})")
    # print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    if (terminated or truncated):
        observation, info = env.reset()
env.close()

