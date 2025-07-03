#!/usr/bin/env python3

import pytest

# BAM

from bam_gym.envs import custom_spaces  
from transforms3d.euler import euler2quat

from ros_py_types.geometry_msgs import PoseStamped, Pose
from ros_py_types.std_msgs import Header
from ros_py_types.sensor_msgs import CameraInfo
from ros_py_types.bam_msgs import Segment2D, Segment2DArray
from ros_py_types.non_ros_msgs import Grasp

# PYTHON
import numpy as np


""" Design Notes:

    Its very important to be able to go back and forth between dictionaries schema, which is what gym spaces, are and the ros messages,
    Which have type hints, helpers, are easy to work with and are ultimately what the BAM GymAPI uses.

    How to pass grasps?
    - as always why not support a couple of ways...

    I don't have a grasp message type, instead I have a list of post_stamped, which is just a classic datastructure to pass around (ie. nav_msgs/Path)
    And then a custom waypoint params that can be used to set a whole number of parameters related to that one point, including grasp width


    When dealing with multiple arrays etc, it is nice to pass in a 7 DOF array thoughts:

    [x, y, z, r, p, y, w]

    I also like this custom grasp space that is easy to parse:

    ```python
    def grasp_space(position_bounds=(-10.0, 10.0), orientation_bounds=(-np.pi, np.pi), grasp_width_bounds=(0.0, 0.2)):

        space = spaces.Dict({
            "pose": pose_stamped_space(position_bounds, orientation_bounds),
            "grasp_width": spaces.Box(
                low=grasp_width_bounds[0],
                high=grasp_width_bounds[1],
                shape=(),
                dtype=np.float32
            ),
        })

        return space
    ```

    In these tests Is should basically make custom spaces, sample them, verify the values load, then go back to dict, and verify key and values are the same.
"""
""" CLI Shortcuts:

pytest -s ~/python_ws/bam_gym/tests/test_custom_spaces.py
"""

def test_obs_space():
    space = custom_spaces.obs_space()

def assert_pose_equal(msg: Pose, msg_dict: dict):
    """ Design Notes:

    1. space is euluer, Pose is a quaternion, so you cannot compare!
    assert msg.orientation.x  == msg_dict["orientation"]["x"]

    2. euluer could be non-unique, so we can't compare!
    array = msg.to_array()
    assert array[3] == msg_dict["orientation"]["x"]
    assert array[4] == msg_dict["orientation"]["y"]
    assert array[5] == msg_dict["orientation"]["z"]

    3. convert to quaternion and compare

    [UPDATE] Now all dict representations should just be quaternion, and all array representations should be euler angles.
    """
    assert msg.position.x == msg_dict["position"]["x"]
    assert msg.position.y == msg_dict["position"]["y"]
    assert msg.position.z == msg_dict["position"]["z"]
    # w, x, y, z = euler2quat(msg_dict["orientation"]["x"], msg_dict["orientation"]["y"], msg_dict["orientation"]["z"])
    # assert np.isclose(x, msg.orientation.x)
    # assert np.isclose(y, msg.orientation.y)
    # assert np.isclose(z, msg.orientation.z)
    # assert np.isclose(w, msg.orientation.w)
    assert msg.orientation.x == msg_dict["orientation"]["x"]
    assert msg.orientation.y == msg_dict["orientation"]["y"]
    assert msg.orientation.z == msg_dict["orientation"]["z"]
    assert msg.orientation.w == msg_dict["orientation"]["w"]

def assert_header_equal(msg: Header, msg_dict: dict):
    assert msg.stamp.sec == msg_dict["stamp"]["sec"]
    assert msg.stamp.nanosec == msg_dict["stamp"]["nanosec"]
    assert msg.frame_id == msg_dict["frame_id"]

def test_pose_space():
    space = custom_spaces.pose_space()
    sample = space.sample()
    assert_pose_equal(Pose.from_dict(sample), sample)

def test_header_space():
    space = custom_spaces.header_space()
    sample = space.sample()
    msg = Header.from_dict(sample)
    assert_header_equal(msg, sample)

def test_pose_stamped_space():
    space = custom_spaces.pose_stamped_space()
    sample = space.sample()
    msg = PoseStamped.from_dict(sample)
    assert_header_equal(msg.header, sample["header"])
    assert_pose_equal(msg.pose, sample["pose"])

def test_camera_info_space():
    space = custom_spaces.camera_info_space()
    sample = space.sample()
    msg = CameraInfo.from_dict(sample)
    assert msg.height == sample["height"]
    assert msg.width == sample["width"]
    assert np.array_equal(msg.k, sample["k"])
    assert np.array_equal(msg.d, sample["d"])


def test_grasp_space():
    space = custom_spaces.grasp_space()
    sample = space.sample()
    msg = Grasp.from_dict(sample)
    assert_header_equal(msg.pose.header, sample["pose"]["header"])
    assert_pose_equal(msg.pose.pose, sample["pose"]["pose"])

# TODO test other spaces...