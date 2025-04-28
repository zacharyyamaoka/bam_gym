#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync") 

observation, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep Result:")
    print(f"Action: {action} (shape: {reward.shape}, type: {type(action)})")
    print(f"Observation: {observation} (shape: {observation.shape}, type: {type(observation)})")
    print(f"Reward: {reward} (shape: {reward.shape}, type: {type(reward)})")
    print(f"Terminated: {terminated} (shape: {terminated.shape}, type: {type(terminated)})")
    print(f"Truncated: {truncated} (shape: {truncated.shape}, type: {type(truncated)})")
    print(f"Info: {info} (type: {type(info)}, length: {len(info)})")

    # Must check per environment if terminated/truncated
    if np.any(terminated) or np.any(truncated):
        observation, info = env.reset()
env.close()