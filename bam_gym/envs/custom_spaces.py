import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Only include values in the spaces that people actually use/need to know about...

#TODO for vector
# self.observation_space = custom_spaces.MaskedSpaceWrapper(self.observation_space, mask=(self.num_envs, self.obs_mask))

class Float(spaces.Space):
    def __init__(self, low=-np.inf, high=np.inf):
        super().__init__(shape=(), dtype=float)
        self.low = low
        self.high = high
    
    def sample(self):
        return float(np.random.uniform(self.low, self.high))
    
    def contains(self, x):
        return isinstance(x, (int, float)) and self.low <= x <= self.high



def bam_obs_space(n_obs = 0, n_color = 0, n_depth = 0, n_pose = 0, n_detection = 0, automask = True):

        obs_dict = spaces.Dict({})
        obs_mask = {} 

        if n_obs:
            obs_dict["obs_names"] = spaces.Sequence(spaces.Text(max_length=32))
            obs_dict["obs"] = spaces.Sequence(spaces.Box(low=-1, high=1, shape=(n_obs,), dtype=np.float32))
            obs_mask["obs_names"] = (n_obs, None)
            obs_mask["obs"] = (n_obs, None)
        if n_color:
            obs_dict["color_img"] = spaces.Sequence(color_img_space())
            obs_mask["color_img"] = (n_color,)

        if n_depth:
            obs_dict["depth"] = spaces.Sequence(depth_img_space())
            obs_mask["depth"] = (n_depth,)

        if n_color or n_depth:
            obs_dict["camera_info"] = spaces.Sequence(camera_info_space())
            obs_mask["camera_info"] = (max(n_color, n_depth),)

        if n_pose:
            obs_dict["pose_names"] = spaces.Sequence(spaces.Text(max_length=32))
            obs_dict["pose"] = spaces.Sequence(pose_stamped_space())
            obs_mask["pose_names"] = (n_pose, None)
            obs_mask["pose"] = (n_pose, None)
        # Add segments to observation space
        if n_detection:
            obs_dict["segments"] = spaces.Sequence(segment2darray_space())
            obs_mask["segments"] = (n_detection, None)

        if automask:
            return MaskedSpaceWrapper(obs_dict, obs_mask)
        else:
            return obs_dict

def manipulator_api_space(n_waypoints=1, moteus_params=False, moveit_params=False, automask = True):

    space = spaces.Dict({})
    space_mask = {}

    space["names"] = spaces.Sequence(spaces.Text(max_length=32))
    space["waypoints"] = spaces.Sequence(pose_stamped_space())
    space["parameters"] = spaces.Sequence(waypoint_params_space(moteus_params, moveit_params))

    space_mask["names"] = (n_waypoints, None)
    space_mask["waypoints"] = (n_waypoints, None)
    space_mask["parameters"] = (n_waypoints, None)

    if automask:
        return MaskedSpaceWrapper(space, space_mask)
    else:
        return space


def waypoint_params_space(moteus_params=False, moveit_params=False):
    """
    Returns a gym space for waypoint parameters including MOTEUS, MOVEIT, and MISC parameters.
    """
    space = spaces.Dict({})

    if moteus_params:
        space["kp_scale"] = Float(low=0.0, high=1.0)  # dynamically adjust stiffness
        space["kd_scale"] = Float(low=0.0, high=1.0)
        space["max_velocity"] = Float(low=0.0, high=100.0)
        space["max_effort"] = Float(low=0.0, high=1000.0)
        
    if moveit_params:
        # MOVEIT parameters
        space["target_link"] = spaces.Text(max_length=32)  # TCP, IK_TIP, etc. what is the target link?
        space["planner"] = spaces.Text(max_length=16)  # available are "ptp", "lin"
        space["vel_scale"] = Float(low=0.0, high=1.0)
        space["accel_scale"] = Float(low=0.0, high=1.0)
        space["blend_radius"] = Float(low=0.0, high=1.0)  # Used to blend in MOVE_TO_SEQUENCE
        space["goal_tol"] = Float(low=0.001, high=0.1)
        space["path_tol"] = Float(low=0.001, high=0.1)
        space["vertical_angle_scale"] = Float(low=0.0, high=1.0)  # 0 (Approach/Retreat completely along grasp z_axis) to 1 (Approach/Retreat offsets should be completely vertical to gravity/table surface)
    
    space["gripper_width"] = Float(low=0.0, high=0.2)
    
    return space


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
            "x": Float(low=pos_low, high=pos_high),
            "y": Float(low=pos_low, high=pos_high),
            "z": Float(low=pos_low, high=pos_high),
        }),
        "orientation": spaces.Dict({
            "x": Float(low=ori_low, high=ori_high),  
            "y": Float(low=ori_low, high=ori_high),  
            "z": Float(low=ori_low, high=ori_high), 
            "w": Float(low=ori_low, high=ori_high), 

        }),
    })

    return space

def header_space():
    stamp_space = spaces.Dict({
        "sec": Float(low=0, high=2**31 - 1),
        "nanosec": Float(low=0, high=1_000_000_000 - 1),
    })

    space = spaces.Dict({
        "stamp": stamp_space,
        "frame_id": spaces.Text(max_length=32),
    })

    return space
    
def pose_stamped_space(
    position_bounds=(-10.0, 10.0),
    orientation_bounds=(-np.pi, np.pi),
):
    """Returns a gym space for geometry_msgs/msg/PoseStamped."""

    pose = pose_space(position_bounds, orientation_bounds)

    space = spaces.Dict({
        "header": header_space(),
        "pose": pose,
    })

    return space

def grasp_space(position_bounds=(-10.0, 10.0), orientation_bounds=(-np.pi, np.pi), grasp_width_bounds=(0.0, 0.2)):
    """
    Returns a gym space for a grasp, which includes a pose and a grasp width.
    """
    space = spaces.Dict({
        "pose": pose_stamped_space(position_bounds, orientation_bounds),
        "gripper_width": Float(
            low=grasp_width_bounds[0],
            high=grasp_width_bounds[1]
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
        "height": Float(low=0, high=10000),
        "width": Float(low=0, high=10000),
        "k": spaces.Box(low=-1e6, high=1e6, shape=(9,), dtype=np.float64),
        "d": spaces.Box(low=-1e6, high=1e6, shape=(8,), dtype=np.float64),
    })

def detection_space(max_id=1e6, max_size=(600, 480), score_range=(0.0, 1.0)):
    """Returns a gym Dict space for Detection2D messages."""
    max_x, max_y = max_size

    space = spaces.Dict({
        "bbox": spaces.Dict({
            "center": spaces.Dict({
                "x": Float(low=0, high=max_x),
                "y": Float(low=0, high=max_y),
            }),
            "size_x": Float(low=0, high=max_x),
            "size_y": Float(low=0, high=max_y),
        }),
        "results": spaces.Sequence(
            spaces.Dict({
                "id": Float(low=0, high=max_id),
                "score": Float(low=score_range[0], high=score_range[1])
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
                "x": Float(low=0, high=max_x),
                "y": Float(low=0, high=max_y),
            })
        )
    })

    return space

def segment2d_space():
    return spaces.Dict({
        "header": header_space(),
        "results": spaces.Sequence(spaces.Dict({})),  # ObjectHypothesisWithPose
        "bbox": spaces.Dict({}),
        "id": spaces.Text(max_length=64),
        "polygon": spaces.Sequence(spaces.Dict({})),  # Polygon
        "mask": spaces.Dict({}),  # Image
        "img_height": Float(low=0, high=2**16-1),
        "img_width": Float(low=0, high=2**16-1),
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



class MaskedSpaceWrapper(spaces.Space):
    """
    Convience class so you don't need to call space.sample(mask=(1,None)), it automatically applies the mask if provided.
    
    """
    def __init__(self, space: spaces.Space, mask=None):
        super().__init__(space.shape if hasattr(space, "shape") else None, space.dtype if hasattr(space, "dtype") else None)
        self.space = space
        self.mask = mask

    def sample(self):
        # Custom sample logic using mask
        if hasattr(self.space, "sample"):
            try:
                return self.space.sample(mask=self.mask)
            except TypeError:
                # fallback for spaces that don't support masks natively
                return self.space.sample()
        raise NotImplementedError("Wrapped space does not support sampling")

    def contains(self, x):
        return self.space.contains(x)

    def __getattr__(self, name):
        # Delegate all other attributes
        return getattr(self.space, name)
    
    def __repr__(self):
        return f"<MaskedSpace<{str(self.space)}>>"