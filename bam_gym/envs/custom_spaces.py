import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Only include values in the spaces that people actually use/need to know about...

def obs_space(value_range=(-np.inf, np.inf), max_obs_dim=1):
    """
    Returns separate gym spaces for obs_names and obs.
    """
    low, high = value_range

    space = spaces.Box(low=low, high=high, shape=(max_obs_dim,), dtype=np.float32)

    return space


def pose_space(position_bounds=(-10.0, 10.0), orientation_bounds=(-np.pi, np.pi)):
    """Returns a gym space for a Pose with position and orientation (Euler). if you convert into a ros_py_type, it will be a quaternion."""
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

def segment2d_space():
    return spaces.Dict({
        "header": spaces.Dict({}),  # You can further specify fields if needed
        "results": spaces.Sequence(spaces.Dict({})),  # ObjectHypothesisWithPose
        "bbox": spaces.Dict({}),
        "id": spaces.Text(max_length=64),
        "polygon": spaces.Sequence(spaces.Dict({})),  # Polygon
        "mask": spaces.Dict({}),  # Image
        "img_height": spaces.Box(low=0, high=2**16-1, shape=(), dtype=int),
        "img_width": spaces.Box(low=0, high=2**16-1, shape=(), dtype=int),
    })

def segment2darray_space():
    return spaces.Dict({
        "header": spaces.Dict({}),
        "segments": spaces.Sequence(segment2d_space()),
    })

def grasp_array_space(
    xyz_bounds=(-10.0, 10.0),
    rpy_bounds=(-np.pi, np.pi),
    grasp_width_bounds=(0.0, 0.1)
):
    """
    Returns a gym Box space for a grasp represented as a 7D array: (x, y, z, rx, ry, rz, w)
    """
    low = np.array([
        xyz_bounds[0],  # x
        xyz_bounds[0],  # y
        xyz_bounds[0],  # z
        rpy_bounds[0],  # rx
        rpy_bounds[0],  # ry
        rpy_bounds[0],  # rz
        grasp_width_bounds[0],  # w
    ], dtype=np.float32)
    high = np.array([
        xyz_bounds[1],  # x
        xyz_bounds[1],  # y
        xyz_bounds[1],  # z
        rpy_bounds[1],  # rx
        rpy_bounds[1],  # ry
        rpy_bounds[1],  # rz
        grasp_width_bounds[1],  # w
    ], dtype=np.float32)
    return spaces.Box(low=low, high=high, shape=(7,), dtype=np.float32)