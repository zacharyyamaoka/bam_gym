import gymnasium as gym
import bam_gym
import numpy as np

# Add new environments here to list out
env_ids = [
    "CartPole-v1",
    "bam/GridWorld-v0",
    "bam/MNIST",
]

for i, env_id in enumerate(env_ids):
    print(f"\n=== Environment {i+1}: {env_id} ===")
    
    env = gym.make(env_id)

    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    action = env.action_space.sample()
    observation, info = env.reset(seed=42)

    print(f"Sampled Action: {action}")
    if isinstance(observation, np.ndarray):
        print(f"  Inital Observation: np.ndarray {observation.shape}")
    else:
        print(f"  Inital Observation: {observation}")
    next_obs, reward, terminated, truncated, info = env.step(action)

    print("Step Result:")
    if isinstance(observation, np.ndarray):
        print(f"  Next Observation: np.ndarray {next_obs.shape}")
    else:
        print(f"  Next Observation: {next_obs}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    print(f"  Info: {info}")

    env.close()
