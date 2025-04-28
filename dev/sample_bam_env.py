#!/usr/bin/env python3

import gymnasium as gym
from bam_gym.envs import CartPole
from bam_gym.ros_types.bam_msgs import ErrorCode, ErrorType
from bam_gym.transport import RoslibpyTransport
import copy


# First make transport, this allows it communicate with backend server
transport = RoslibpyTransport("test_ns")

# env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("bam/GridWorld-v0")
# env = gym.make("bam/MNIST", render_mode="human")
# env = gym.make("bam/CartPole", transport=transport, render_mode="human")
env = CartPole(transport=transport, render_mode="human")

# env = gym.make("bam/GraspV1", transport=transport, render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample(mask=(1,None))
    observation, reward, terminated, truncated, info = env.step(action)

    # Bam

    display_obs = copy.deepcopy(observation)
    for obs in display_obs:
        if "color_img" in obs:
            obs["color_img"] = f"np.ndarray{obs['color_img'].shape}"

        if "depth_img" in obs:
            obs["depth_img"] = f"np.ndarray{obs['depth_img'].shape}"

    print(f"\nStep Result:")
    print(f"Action: {action}")
    print(f"Observation: {display_obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    display_info = copy.deepcopy(info)
    display_info["header"]["error_code"]["value"] = ErrorType(display_info["header"]["error_code"]["value"]).name
    print(f"Info: {display_info}")


    # Handle error:
    # Simulated environments always return observations, but sometimes
    # real environments have issues! need to access full info and check if success
    if info["header"]["error_code"]["value"] != ErrorType.SUCCESS:
        print("Skipping this step due to error.")
        print("Error message:", info["header"].get("error_msg", ""))
        continue

    # No need to reset as env auto resets
    if True and (terminated[0] or truncated[0]):
        observation, info = env.reset()

env.close()

