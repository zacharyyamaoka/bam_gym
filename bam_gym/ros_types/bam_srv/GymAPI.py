from enum import IntEnum
from typing import List, Optional, Dict, Any
import numpy as np
import copy
import json

from bam_gym.ros_types.bam_msgs import RequestHeader, ResponseHeader, GymAction, GymFeedback, ErrorType

class RequestType(IntEnum):
    NONE = 0
    STEP = 1
    RESET = 2
    CLOSE = 3

class GymAPIRequest:
    def __init__(self):
        self.header = RequestHeader()
        self.seed: int = 0
        self.env_name = ""
        self.action: List[GymAction] = []

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "seed": self.seed,
            "env_name": self.env_name,
            "action": [a.to_dict() for a in self.action],
        }

class GymAPIResponse:
    def __init__(self):
        self.header = ResponseHeader()
        self.feedback: List[GymFeedback] = []

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "feedback": [f.to_dict() for f in self.feedback],
        }
    
    def to_step_tuple(self):
        """Unpack all responses into separate lists."""
        observations = []
        rewards = []
        terminated = []
        truncated = []
        infos = {}

        for idx, f in enumerate(self.feedback):
            obs, reward, term, trunc, info = f.to_step_tuple()
            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos[idx] = info

        return (
            np.array(observations),                # (N, obs_dim)
            np.array(rewards, dtype=np.float32),    # (N,)
            np.array(terminated, dtype=bool),       # (N,)
            np.array(truncated, dtype=bool),        # (N,)
            infos    
        )

    def to_reset_tuple(self):
        """Unpack reset responses into separate lists."""
        observations = []
        infos = {}

        for idx, f in enumerate(self.feedback):
            obs, info = f.to_reset_tuple()
            observations.append(obs)
            infos[idx] = info

        return (
            np.array(observations),  # shape (N, obs_dim)
            infos                    # list of dicts
        )
    
    @classmethod
    def from_dict(cls, d: dict):
        print("GYM API FROM DICT")
        obj = cls()
        obj.header = ResponseHeader.from_dict(d.get("header", {}))
        obj.feedback = [GymFeedback.from_dict(f) for f in d.get("feedback", [])]
        return obj

    def __str__(self):

        display_response = copy.deepcopy(self.to_dict())

        # Handle header.error_code nicely
        try:
            error_value = display_response["header"]["error_code"].get("value", 0)
            display_response["header"]["error_code"]["value"] = ErrorType(error_value).name
        except Exception as e:
            print(f"Warning converting error code: {e}")

        # Handle feedback images nicely
        for f in display_response.get("feedback", []):
            if isinstance(f.get("color_img"), dict) and "shape" in f["color_img"]:
                f["color_img"] = f"np.ndarray{tuple(f['color_img']['shape'])}"
            elif hasattr(f.get("color_img"), "shape"):
                f["color_img"] = f"np.ndarray{f['color_img'].shape}"

            if isinstance(f.get("depth_img"), dict) and "shape" in f["depth_img"]:
                f["depth_img"] = f"np.ndarray{tuple(f['depth_img']['shape'])}"
            elif hasattr(f.get("depth_img"), "shape"):
                f["depth_img"] = f"np.ndarray{f['depth_img'].shape}"

        return json.dumps(display_response, indent=2)