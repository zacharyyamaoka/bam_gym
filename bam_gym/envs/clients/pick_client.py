# BAM
from bam_gym.envs.clients.generic_gym_client import GenericGymClient
from bam_gym.transport import RoslibpyTransport, MockTransport
from ros_py_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from ros_py_types.bam_msgs import ErrorCode, GymAction, GymFeedback
from ros_py_types.geometry_msgs import PoseStamped, Pose
from bam_gym.utils.parameters import get_default_params

from bam_gym.utils.utils import ensure_list

from bam_gym.envs import custom_spaces

# PYTHON
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from typing import Any, SupportsFloat, Tuple, List, Dict
"""


Vs Pick and Lift, this is faster, as soon as the pick attempts finishes, the object spawns in a new static location
"""

class PickClient(GenericGymClient):

    def __init__(self, transport = MockTransport(), n_pose=1, **kwargs):
        
        super().__init__(transport=transport, n_pose=n_pose,  **kwargs)

        self.env_name = "Pick"
        self.pick_params = get_default_params()
        self.action_space = custom_spaces.grasp_space()

        if self.vec:
            self.single_action_space = self.action_space
            self.action_space = spaces.Sequence(self.single_action_space)

    def _init_action_space(self):
        self.single_action_space = custom_spaces.grasp_space()
        self.action_space = spaces.Sequence(self.single_action_space)
    
    def step(self, action) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:

        # Convert from action to GymAPI_Request
        request = GymAPI_Request()

        action_msg_list =[]

        # convert gym action type to dummy ros action type
        if not isinstance(action, (list, tuple)):
            action = [action]

        for a in action:
            action_msg = GymAction()

            pose_stamped = PoseStamped.from_dict(a['pose'])
            print(pose_stamped.to_dict())

            gripper_width = a['gripper_width']
            self.pick_params.gripper_width = gripper_width

            action_msg.pose_names = ['pick']
            action_msg.pose_action = [pose_stamped]
            action_msg.pose_param = [self.pick_params]

            action_msg_list.append(action_msg)

        print(f"Sending {len(action_msg_list)} actions")
        request.action = action_msg_list

        # Get response
        response: GymAPI_Response = self._step(request)

        # Convert from GymAPI_Response to (observation, reward, terminated, truncated, info)
        (observations, rewards, terminated, truncated, infos) = response.to_step_tuple()

        self._render()

        return (observations, rewards, terminated, truncated, infos) # type: ignore
    
    def render(self):
        return self._render()

    def close(self) -> GymAPI_Response:
        return self._close()


if __name__ == '__main__':
    from bam_gym import print_gym_space, print_action
    env = PickClient()

    print(type(env.action_space.sample()))
    
    print("\nACTION SPACE: \n")
    print_gym_space(env.action_space)

    print("\nOBSERVATION SPACE: \n", )
    print_gym_space(env.observation_space)

    action = env.action_space.sample()
    print("\nSAMPLED ACTION: \n", )
    print_action(action)

    # (observations, infos) = env.reset()
    # assert env.success
    # print(observations, infos)
    # action = env.action_space.sample(mask=(1,None))
    # (observations, rewards, terminated, truncated, infos) = env.step(action)
    # assert env.success
    # print(observations, rewards, terminated, truncated, infos)