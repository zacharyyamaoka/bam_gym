#!/usr/bin/env python3

import gymnasium as gym
from bam_gym.envs import CartPole
from ros_py_types.bam_msgs import ErrorCode, ErrorType
from bam_gym.transport import RoslibpyTransport
import copy

"""
Bam Environments 
(sequences can be variable length, but will all be the same length)

Actions: Sequences of Actions (1 per arm, for retries, for different robots, etc.)

Observations: Sequence of Observation, duplicate observation will be none, check the info dict

Reward: Sequence of rewards. Will be 0 in case action is exectued=False

Terminate: Sequence of terminated signal. For 1 step environments, don't pay attention, just manually terminate, 

Info: Keys of '0','1','2', etc. for info regarding action/observation index. Key 'header' for meta info

    - 'executed' - False if action didn't execute, disregard it
    - 'duplicate_obs_ns' - True if the obs is the same as another one already in the list
    - 'namespace' - the rack namespace this feedback belongs to

--
Design Notes:

- This works nicely with the GymAPI, that accepts lists of actions returns list of feedback
- This is a very generic API that should cover all basis!
"""
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

