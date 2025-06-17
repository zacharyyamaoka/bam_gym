
import gymnasium as gym
from gymnasium import spaces
import numpy as np


def obs_space(value_range=(-np.inf, np.inf), max_obs_dim=1):
    """
    Returns separate gym spaces for obs_names and obs.
    """
    low, high = value_range

    space = spaces.Box(low=low, high=high, shape=(max_obs_dim,), dtype=np.float32)

    return space

def pose_space(position_bounds=(-10.0, 10.0), orientation_bounds=(-np.pi, np.pi)):
    """Returns a gym space for a Pose with position and orientation (Euler)."""
    pos_low, pos_high = position_bounds
    ori_low, ori_high = orientation_bounds

    space = spaces.Dict({
        "position": spaces.Dict({
            "x": spaces.Box(low=pos_low, high=pos_high, shape=(), dtype=np.float32),
            "y": spaces.Box(low=pos_low, high=pos_high, shape=(), dtype=np.float32),
            "z": spaces.Box(low=pos_low, high=pos_high, shape=(), dtype=np.float32),
        }),
        "orientation": spaces.Dict({
            "x": spaces.Box(low=ori_low, high=ori_high, shape=(), dtype=np.float32),  # roll
            "y": spaces.Box(low=ori_low, high=ori_high, shape=(), dtype=np.float32),  # pitch
            "z": spaces.Box(low=ori_low, high=ori_high, shape=(), dtype=np.float32),  # yaw
        }),
    })

    return space

def pose_stamped_space(
    position_bounds=(-10.0, 10.0),
    orientation_bounds=(-np.pi, np.pi),
):
    """Returns a gym space for geometry_msgs/msg/PoseStamped."""

    stamp_space = spaces.Dict({
        "sec": spaces.Box(low=0, high=2**31 - 1, shape=(), dtype=np.int32),
        "nanosec": spaces.Box(low=0, high=1_000_000_000 - 1, shape=(), dtype=np.int32),
    })

    header_space = spaces.Dict({
        # "stamp": stamp_space,
        "frame_id": spaces.Text(max_length=32),
    })

    pose = pose_space(position_bounds, orientation_bounds)

    space = spaces.Dict({
        "header": header_space,
        "pose": pose,
    })

    return space

def grasp_space(position_bounds=(-10.0, 10.0), orientation_bounds=(-np.pi, np.pi), grasp_width_bounds=(0.0, 0.2)):
    """
    Returns a gym space for a grasp, which includes a pose and a grasp width.
    """
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

# OLD - keep as full json dict so that to_dict() and from_dict() work for both
# def pose_space():

#     space = spaces.Dict({
#         "position": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),  # x, y, z
#         "orientation": spaces.Box(low=-3.14, high=3.14, shape=(3,), dtype=np.float32),  # euler x, y, z
#         # "header": spaces.Dict({  # Optional, if needed
#         #     "frame_id": spaces.Text(max_length=100),  # Or just skip it
#         #     "stamp": spaces.Box(low=0, high=1e9, shape=(1,), dtype=np.float64),
#         # }),
#     })

#     return space

def color_img_space(shape=(480, 600, 3), dtype=np.uint8, value_range=(0, 255)):
    """Returns a gym Box space for an RGB image."""
    low, high = value_range

    space = spaces.Box(
        low=low,
        high=high,
        shape=shape,
        dtype=dtype
    )

    return space

def depth_img_space(shape=(480, 600), dtype=np.uint16, value_range=(0, 65535)):
    """Returns a gym Box space for a depth image."""
    low, high = value_range

    space = spaces.Box(
        low=low,
        high=high,
        shape=shape,
        dtype=dtype
    )

    return space

def camera_info_space():
    return spaces.Dict({
        "height": spaces.Box(low=0, high=10000, shape=(), dtype=np.int32),
        "width": spaces.Box(low=0, high=10000, shape=(), dtype=np.int32),
        "k": spaces.Box(low=-1e6, high=1e6, shape=(9,), dtype=np.float64),
        "d": spaces.Box(low=-1e6, high=1e6, shape=(8,), dtype=np.float64),
    })

def detection_space(max_id=1e6, max_size=(600, 480), score_range=(0.0, 1.0)):
    """Returns a gym Dict space for Detection2D messages."""
    max_x, max_y = max_size

    space = spaces.Dict({
        "bbox": spaces.Dict({
            "center": spaces.Dict({
                "x": spaces.Box(low=0, high=max_x, shape=(), dtype=np.float32),
                "y": spaces.Box(low=0, high=max_y, shape=(), dtype=np.float32),
            }),
            "size_x": spaces.Box(low=0, high=max_x, shape=(), dtype=np.float32),
            "size_y": spaces.Box(low=0, high=max_y, shape=(), dtype=np.float32),
        }),
        "results": spaces.Sequence(
            spaces.Dict({
                "id": spaces.Box(low=0, high=max_id, shape=(), dtype=np.int64),
                "score": spaces.Box(low=score_range[0], high=score_range[1], shape=(), dtype=np.float32)
            })
        )
    })

    return space

def mask_space(max_size=(600, 480)):
    """Returns a gym Dict space for polygon masks."""
    max_x, max_y = max_size

    space = spaces.Dict({
        "points": spaces.Sequence(
            spaces.Dict({
                "x": spaces.Box(low=0, high=max_x, shape=(), dtype=np.float32),
                "y": spaces.Box(low=0, high=max_y, shape=(), dtype=np.float32),
            })
        )
    })

    return space