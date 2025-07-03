#!/usr/bin/env python3

import pytest

# BAM


import bam_gym
from bam_gym.wrappers import MockEnv, MockObs
from bam_gym.transport import MockTransport
from bam_gym.utils.pprint import print_action
from bam_gym.runners import run_agent
from ros_py_types.non_ros_msgs import Grasp
from ros_py_types.geometry_msgs import PoseStamped, Pose
from ros_py_types.std_msgs import Header

# PYTHON
import numpy as np
import gymnasium as gym


""" Design Notes:

How do I know the blind policy is working?
- if the env doesn't have pose, it should throw an error
- if the env has pose, it should return a grasp, or list of grasp (self.vec = True)

"""
""" CLI Shortcuts:

pytest -s ~/python_ws/bam_gym/tests/test_custom_spaces.py
"""
def assert_pose_equal(msg_1: Pose, msg_2: Pose):
    assert np.isclose(msg_1.position.x, msg_2.position.x)
    assert np.isclose(msg_1.position.y, msg_2.position.y)
    assert np.isclose(msg_1.position.z, msg_2.position.z)
    assert np.isclose(msg_1.orientation.x, msg_2.orientation.x)
    assert np.isclose(msg_1.orientation.y, msg_2.orientation.y)
    assert np.isclose(msg_1.orientation.z, msg_2.orientation.z)
    assert np.isclose(msg_1.orientation.w, msg_2.orientation.w)

def assert_header_equal(msg_1: Header, msg_2: Header):
    assert np.isclose(msg_1.stamp.sec, msg_2.stamp.sec)
    assert np.isclose(msg_1.stamp.nanosec, msg_2.stamp.nanosec)
    assert msg_1.frame_id == msg_2.frame_id

def assert_pose_stamped_equal(msg_1: PoseStamped, msg_2: PoseStamped):
    assert_header_equal(msg_1.header, msg_2.header)
    assert_pose_equal(msg_1.pose, msg_2.pose)


def test_blind_policy_no_pose():
    env = MockObs(gym.make('bam/GenericGymClient', disable_env_checker=True, n_pose=0))
    policy = bam_gym.make_policy('BlindPolicy')
    with pytest.raises(ValueError):
        policy._validate_environment(env)

def test_blind_policy_one_pose():
    env = MockObs(gym.make('bam/GenericGymClient', disable_env_checker=True, n_pose=1))
    policy = bam_gym.make_policy('BlindPolicy')
    policy._validate_environment(env)

    obs, info = env.reset()
    reward, terminated, truncated = None, None, None

    for i in range(10):
        action, action_info = policy(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)


def test_blind_policy_four_pose():
    env = MockObs(gym.make('bam/GenericGymClient', disable_env_checker=True, n_pose=4))
    policy = bam_gym.make_policy('BlindPolicy')
    policy._validate_environment(env)

@pytest.mark.parametrize("env_name", ["bam/GenericGymClient", "bam/PickClient"])
def test_blind_policy_pick_env(env_name):
    env = MockObs(gym.make(env_name, disable_env_checker=True, n_pose=1))
    policy = bam_gym.make_policy('BlindPolicy')
    policy._validate_environment(env)

    obs, info = env.reset()
    reward, terminated, truncated = None, None, None

    # ok so right now its just using a default grasp width, beacuse there is no grasp obs space. I think its ok tbh...
    for i in range(10):
        obs_pose_stamped = PoseStamped.from_dict(obs['pose'][0]) # take obs before step()
        action, action_info = policy(obs, reward, terminated, truncated, info)
        grasp = Grasp.from_dict(action) # works with PoseStamped and Grasp Dicts!
        grasp_pose_stamped = grasp.pose
        assert_pose_stamped_equal(grasp_pose_stamped, obs_pose_stamped) #NOTE this verifies that the header frame_id is the same!

        obs, reward, terminated, truncated, info = env.step(action)
        # print(f"\n {i}")
        # print(grasp_pose_stamped)
        # print(obs_pose_stamped)

@pytest.mark.parametrize("env_name", ["bam/GenericGymClient", "bam/PickClient"])

def test_blind_policy_run_agent(env_name):
    env = MockObs(gym.make(env_name, disable_env_checker=True, n_pose=1))
    policy = bam_gym.make_policy('BlindPolicy')
    run_agent(policy, env, n_steps=10)

