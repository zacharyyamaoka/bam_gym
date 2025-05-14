#!/usr/bin/env python3

import gymnasium as gym
from bam_gym.envs import GraspXYR
from bam_gym.ros_types.bam_msgs import ErrorCode, ErrorType
from bam_gym.transport import RoslibpyTransport
from bam_gym.utils import print_step_result


transport = RoslibpyTransport("bam_GPU")
env = GraspXYR(transport=transport, render_mode="human")

print(env.action_space)
print(env.observation_space)

print("Sampling 1: ", env.action_space.sample(mask=(1,None)))
print("Sampling 4: ", env.action_space.sample(mask=(4,None)))

observation, info = env.reset(seed=42)


# for _ in range(100):
#     action = env.action_space.sample(mask=(1,None)) # Mask sequence to len(1)
#     new_observation, reward, terminated, truncated, info = env.step(action)

#     print_step_result(observation, action, reward, terminated, truncated, info)
#     observation = new_observation

#     # Handle error - Simulated environments always return observations, but sometimes
#     # real environments have issues! need to access full info and check if success
#     if info["header"]["error_code"]["value"] != ErrorType.SUCCESS:
#         print("Skipping this step due to error.")
#         print("Error message:", info["header"].get("error_msg", ""))
#         continue

#     # No need to reset as env auto resets
#     if True and (terminated[0] or truncated[0]):
#         observation, info = env.reset()



env.close()