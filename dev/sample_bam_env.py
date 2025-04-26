#!/usr/bin/env python3

import gymnasium as gym
import bam_gym
from bam_gym.ros_types.bam_msgs import ErrorCode
from bam_gym.transport import RoslibpyTransport
import copy


# First make transport, this allows it communicate with backend server
transport = RoslibpyTransport("bam_GPU")

# env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("bam/GridWorld-v0")
# env = gym.make("bam/MNIST", render_mode="human")
env = gym.make("bam/CartPole", transport=transport, render_mode="human")
# env = gym.make("bam/GraspV1", transport=transport, render_mode="human")

reset_on_terminate = True

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
    
    # Copy to avoid editingn actual info,
    # Replace images, for compact printing and add readable name for ErrorCode
    display_info = copy.deepcopy(info)
    if hasattr(display_info.get("color_img"), "shape"):
        display_info["color_img"] = f"np.ndarray{display_info['color_img'].shape}"
    if hasattr(display_info.get("depth_img"), "shape"):
        display_info["depth_img"] = f"np.ndarray{display_info['depth_img'].shape}"
    display_info["header"]["error_code"]["value"] = ErrorCode.name(display_info["header"]["error_code"]["value"])
    
    print(f"Info: {display_info}")


    # Handle error:
    # Simulated environments always return observations, but sometimes
    # real environments have issues! need to access full info and check if success
    if info["header"]["error_code"]["value"] != ErrorCode.SUCCESS:
        print("Skipping this step due to error.")
        print("Error message:", info["header"].get("error_msg", ""))
        continue

    if (terminated or truncated) and reset_on_terminate:
        print("Calling Reset")
        observation, info = env.reset()

env.close()

