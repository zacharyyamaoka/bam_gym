

# PYTHON
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

# BAM
from bam_gym.envs.remote.bam_env import BamEnv
from bam_gym.transport import RoslibpyTransport, MockTransport
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode, GymAction, GymFeedback
from bam_gym.utils.utils import ensure_list


class ObsEnv(BamEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, transport, render_mode=None):
            
        super().__init__(transport, render_mode=render_mode)

        self.env_name = "ObsEnv"

        self.action_space = spaces.Sequence(
            spaces.Discrete(2)  # Each action is still "Left" (0) or "Right" (1)
        )

    def step(self, action):

        # Convert from action to GymAPI_Request
        request = GymAPI_Request()

        response: GymAPI_Response = self._step(request)

        (observations, rewards, terminated, truncated, infos) = response.to_step_tuple()

        self._render()

        return (observations, rewards, terminated, truncated, infos)
    

if __name__ == "__main__":
    # Example usage
    transport = MockTransport(namespace="bam_GPU")
    env = ObsEnv(transport=transport, render_mode="human")

    print(env.action_space)
    print(env.observation_space)

    observation, info = env.reset(seed=42)

    # for _ in range(100):
    #     action = env.action_space.sample(mask=(1, None))  # Mask sequence to len(1)
    #     new_observation, reward, terminated, truncated, info = env.step(action)

    #     print(f"Step {_}: Observation: {new_observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
    #     observation = new_observation

    # env.close()