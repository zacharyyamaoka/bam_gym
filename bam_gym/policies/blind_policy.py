#!/usr/bin/env python3

"""
Blind Policy is used when testing with hardcoded poses sent from simulation.

It doesn't do anything it just reads the first pose that is sent and then returns it.

To make sure this works.

1. The env observation_space should have a "pose" in it
2. It should read the pose and return it as an action, keeping the orginal header, etc.

One challenge is that sometimes you want to return an array pose beacuse thats simpler, sometimes a header dict pose beacuse that is easier
or a bam_response if you already have that. I think I should make it accept interchangable, and just convert automatically.


- [ ] Check how the generic gym env observation space looks/works

"""

from bam_gym.policies.generic_policy import GenericPolicy
from typing import Any, Dict, Tuple
import gymnasium as gym
from bam_gym.envs.clients import GenericGymClient
from ros_py_types.non_ros_msgs import Grasp
from bam_gym.utils.pprint import print_action

class BlindPolicy(GenericPolicy):
    """
    Simple random policy that samples actions from the environment's action space.
    
    This is useful for baseline testing and exploration.
    """
    def __init__(self):
        super().__init__()
        
        self.last_observation: Any = None
        self.step_count: int = 0
        # print("Creating BlindPolicy, this reads the pose from the observation space and returns it as a grasp action Grasp.from_dict(pose, grasp_width=0.05)")

    def _validate_environment(self, env: gym.Env) -> None:
        # Should work with vectorized and non vectorized envs
        # print(env)
        # print(env.unwrapped)
        # print(env.observation_space)
        
        # Get the unwrapped environment to access custom attributes
        unwrapped_env = getattr(env, 'unwrapped', env)
        self.num_envs = getattr(unwrapped_env, 'num_envs', 1)
        
        env.observation_space.seed(1) # reduce flakiness
        obs = env.observation_space.sample()
        
        if self.num_envs > 1:
            pose_list = obs[0].get('pose', ())
        else:
            pose_list = obs.get('pose', ())

        if len(pose_list) < 1: 
            raise ValueError(f"BlindPolicy requires at least 1 pose in the observation space, currently it has: {obs}")
            
    def step(self, observation, reward=None, terminated=None, truncated=None, info=None) -> Tuple[Any, Dict[str, Any]]:

        """
        You need to read the first pose from GymFeedback:
            self.pose_names: List[str] = []
            self.pose: List[PoseStamped] = []

        And send back a grasp in GymAction:
            self.pose_names: List[str] = []
            self.pose_action: List[PoseStamped] = []
            self.pose_param: List[WaypointParams] = []

        The env will take care of fill in the action, you need to return the posestamped and a grasp width.
        """
        # assume that you will always have a least 1 pose, error checking should happen lower down
        if self.num_envs > 1:
            # Vectorized case: return list of poses, one for each environment
            actions = []
            for i in range(self.num_envs):
                pose_list = observation[i].get('pose', ())
                pose_stamped_dict = pose_list[0]
                grasp_dict = dict()
                grasp_dict['pose'] = pose_stamped_dict
                grasp_dict["gripper_width"] = 0.05
                actions.append(grasp_dict)
            return actions, {}
        else:
            # Single environment case: return single pose
            pose_stamped_list = observation.get('pose', ())
            pose_stamped_dict = pose_stamped_list[0]
            grasp_dict = dict()
            grasp_dict['pose'] = pose_stamped_dict
            grasp_dict["gripper_width"] = 0.05
            # grasp = Grasp.from_dict(grasp_dict)
            #  I also need to send a grasp width...
            return grasp_dict, {}  #no need to convert, env should accept an action in the same format
