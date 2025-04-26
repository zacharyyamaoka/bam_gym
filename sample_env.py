#!/usr/bin/env python3

import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")


observation, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep Result:")
    print(f"Action: {action}")
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    if (terminated or truncated):
        observation, info = env.reset()
env.close()