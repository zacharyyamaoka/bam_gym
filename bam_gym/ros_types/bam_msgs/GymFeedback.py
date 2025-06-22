from typing import List
import numpy as np
import json
import copy

from bam_gym.ros_types.geometry_msgs.Polygon import Polygon
from bam_gym.ros_types.geometry_msgs.PoseStamped import PoseStamped

from bam_gym.ros_types.vision_msgs.Detection2D import Detection2D
from bam_gym.ros_types.sensor_msgs.Image import Image
from bam_gym.ros_types.sensor_msgs.CompressedImage import CompressedImage
from bam_gym.ros_types.sensor_msgs.CameraInfo import CameraInfo
from bam_gym.ros_types.bam_msgs.Segment2DArray import Segment2DArray

class GymFeedback:
    def __init__(self):
        self.ns = ""
        self.observation_names: List[str] = []
        self.observation: List[float] = []

        self.pose_names: List[str] = []
        self.pose: List[PoseStamped] = []

        self.target_class: List[int] = []

        self.color_img: List[CompressedImage] = []
        self.depth_img: List[Image] = []
        self.camera_info: List[CameraInfo] = []
        self.segments: List[Segment2DArray] = []

        self.duplicate_obs = False
        self.duplicate_index: int = 0

        self.executed = False
        self.reward: float = 0.0
        self.terminated = False
        self.truncated = False
        self.info = ""

    def to_dict(self):
        return {
            "ns": self.ns,
            "observation_names": self.observation_names,
            "observation": self.observation,

            "pose_names": self.pose_names,
            "pose": [p.to_dict() for p in self.pose],

            "target_class": self.target_class,

            "color_img": [img.to_dict() for img in self.color_img],
            "depth_img": [img.to_dict() for img in self.depth_img],
            "camera_info": [ci.to_dict() for ci in self.camera_info],
            "segments": [seg.to_dict() for seg in self.segments],

            "duplicate_obs": self.duplicate_obs,
            "duplicate_index": self.duplicate_index,

            "executed": self.executed,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }
    
    # This is used inside roslibpy_transport.py to convert json into this type
    # TODO I will need to implement the from_dict functions for pose, color_img etc.
    @classmethod
    def from_dict(cls, d: dict): # from json dict of GymFeedback.msg
        obj = cls()
        obj.ns = d.get("ns", "")
        obj.observation_names = d.get("observation_names", [])
        obj.observation = d.get("observation", [])

        obj.pose_names = d.get("pose_names", [])
        obj.pose = [PoseStamped.from_dict(p) for p in d.get("pose", [])]
        obj.target_class = d.get("target_class", [])

        obj.color_img = [CompressedImage.from_dict(img) for img in d.get("color_img", [])]
        obj.depth_img = [Image.from_dict(img) for img in d.get("depth_img", [])]
        obj.camera_info = [CameraInfo.from_dict(ci) for ci in d.get("camera_info", [])]
        obj.segments = [Segment2DArray.from_dict(seg) for seg in d.get("segments", [])]

        obj.duplicate_obs = d.get("duplicate_obs", False)
        obj.duplicate_index = d.get("duplicate_index", 0)

        obj.executed = d.get("executed", False)
        obj.reward = d.get("reward", 0.0)
        obj.terminated = d.get("terminated", False)
        obj.truncated = d.get("truncated", False)
        obj.info = d.get("info", "{}")

        return obj

    def obs_dict(self):
        """
        Design Notes:

        - right now I prefer to return the actual types, althought they don't align with the observation space,
        - it gives type completion, which I think will be more helpful!
        """
        obs_dict = {}
        if self.observation:
            obs_dict["obs_names"] = self.observation_names 
            obs_dict["obs"] = np.array(self.observation, dtype=np.float32)

        if self.pose:
            obs_dict["pose_names"] = self.pose_names
            obs_dict["pose"] = [pose_stamped.to_dict() for pose_stamped in self.pose]
            # obs_dict["pose"] = [pose_stamped.pose for pose_stamped in self.pose] I want the pose stamped to send as the action
            # obs_dict["pose"] = [pose_stamped.pose.to_dict() for pose_stamped in self.pose]
            # obs_dict["pose"] = self.pose # to return pose stamped....

        if isinstance(self.color_img, list) and all(isinstance(img, np.ndarray) for img in self.color_img):
            # obs_dict["color_img"] = self.color_img
            obs_dict["color_img"] = [img.data for img in self.color_img]
            # obs_dict["color_img"] = self.color_img.to_dict()

        if isinstance(self.depth_img, list) and all(isinstance(img, np.ndarray) for img in self.depth_img):
            obs_dict["depth_img"] = [img.data for img in self.depth_img]
            # obs_dict["depth_img"] = self.depth_img.to_dict()

        if self.camera_info is not None:
            # obs_dict["camera_info"] = self.camera_info.to_dict()
            obs_dict["camera_info"] = self.camera_info

        return obs_dict
    
    def to_step_tuple(self):

        return (
            self.obs_dict(),
            self.reward,
            self.terminated,
            self.truncated,
            self.info_dump()
        )

    def to_reset_tuple(self):
        return (
            self.obs_dict(),
            self.info_dump()
        )

    def info_dump(self):
        try:
            info_dict = json.loads(self.info) if self.info else {}
        except json.JSONDecodeError as e:
            print("Warning: Failed to parse info JSON:", e)
            info_dict = {}

        # add other values here that may be useful to know!
        # and are not returned by default in observation, reward, etc..
        # generally you want to make all variables accesible.... 

        info_dict['ns'] = self.ns
        info_dict['duplicate_obs'] = self.duplicate_obs
        info_dict['duplicate_index'] = self.duplicate_index
        info_dict['target_class'] = self.target_class

        info_dict['executed'] = self.executed

        return info_dict
    

    def __str__(self):

        display_feedback = copy.deepcopy(self.to_dict())

        # Replace color_img and depth_img with shape info
        if isinstance(self.color_img, list) and all(isinstance(img, np.ndarray) for img in self.color_img):
            display_feedback["color_img"] = [f"np.ndarray{img.data.shape}" for img in self.color_img]

        if isinstance(self.depth_img, list) and all(isinstance(img, np.ndarray) for img in self.depth_img):
            display_feedback["depth_img"] = [f"np.ndarray{img.data.shape}" for img in self.depth_img]


        return json.dumps(display_feedback, indent=2)