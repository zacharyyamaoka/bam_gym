


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

from typing import Tuple, List, Dict
"""

Send a list of pose stamped for different objects to pick up

"""

class PickAndLift(GenericGymClient):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, transport, render_mode=None, plugin = 'gazebo', **kwargs):
            
        super().__init__(transport,
                         obs = False,
                         color = False,
                         depth = False,
                         detections = False,
                         pose = True,
                         render_mode = render_mode
                         )

        self.env_name = "PickAndLift"
        self.pick_params = get_default_params()

        self.action_space = spaces.Sequence(custom_spaces.grasp_space())
            
        
    def reset(self, seed=None) -> Tuple[List, Dict]:

        # Get GymAPI_Response from reset()
        response: GymAPI_Response = self._reset(seed)

        # Convert to (observation, info)
        (observations, infos) = response.to_reset_tuple()

        self._render() # checks internally for render modes

        return (observations, infos)
    
    def step(self, action: spaces.Sequence) -> Tuple[List, List, List, List, Dict]:

        # Convert from action to GymAPI_Request
        request = GymAPI_Request()

        request.header.request_type = RequestType.STEP
        request.env_name = self.env_name

        action_msg_list =[]

        # convert gym action type to dummy ros action type

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

        return (observations, rewards, terminated, truncated, infos)
    
    def render(self):
        return self._render(self)

    def close(self) -> GymAPI_Response:
        return self._close()


if __name__ == '__main__':
    from bam_gym.transport import MockTransport
    from bam_gym.utils.pprint import print_sampled_action, print_gym_space
    env = PickAndLift(MockTransport())

    print(type(env.action_space.sample(mask=(1,None))[0]))
    
    print("\nACTION SPACE: \n")
    print_gym_space(env.action_space)

    print("\nOBSERVATION SPACE: \n", )
    print_gym_space(env.observation_space)

    action = env.action_space.sample(mask=(2,None))
    print("\nSAMPLED ACTION: \n", )
    print_sampled_action(action)

    # (observations, infos) = env.reset()
    # assert env.success
    # print(observations, infos)
    # action = env.action_space.sample(mask=(1,None))
    # (observations, rewards, terminated, truncated, infos) = env.step(action)
    # assert env.success
    # print(observations, rewards, terminated, truncated, infos)