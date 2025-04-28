from typing import List
import numpy as np
import json
import copy

class GymFeedback:
    def __init__(self):
        self.ns = ""
        self.observation: List[float] = []
        self.color_img = None
        self.depth_img = None
        self.reward: float = 0.0
        self.duplicate_obs_ns = False
        self.executed = False
        self.terminated = False
        self.truncated = False
        self.info = ""

    def to_dict(self):
        return {
            "ns": self.ns,
            "observation": self.observation,
            "color_img": self.color_img,
            "depth_img": self.depth_img,
            "reward": self.reward,
            "copy_obs_from_ns": self.copy_obs_from_ns,
            "executed": self.executed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }
    

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls()
        obj.ns = d.get("ns", "")
        obj.observation = d.get("observation", [])
        obj.color_img = d.get("color_img", None)
        obj.depth_img = d.get("depth_img", None)
        obj.reward = d.get("reward", [])
        obj.copy_obs_from_ns = d.get("copy_obs_from_ns", False)
        obj.executed = d.get("executed", False)
        obj.terminated = d.get("terminated", False)
        obj.truncated = d.get("truncated", False)
        obj.info = d.get("info", "{}")
        return obj

    def to_step_tuple(self):
        return (
            np.array(self.observation, dtype=np.float32),
            np.array(self.reward, dtype=np.float32),
            self.terminated,
            self.truncated,
            self.info_dump()
        )

    def to_reset_tuple(self):
        return (
            np.array(self.observation, dtype=np.float32),
            self.info_dump()
        )

    def info_dump(self):
        try:
            info_dict = json.loads(self.info) if self.info else {}
        except json.JSONDecodeError as e:
            print("Warning: Failed to parse info JSON:", e)
            info_dict = {}

        info_dict['ns'] = self.ns
        info_dict['copy_obs_from_ns'] = self.copy_obs_from_ns
        info_dict['executed'] = self.executed


        # if it has been formated to numpy array, return it, otherwise return None
        if hasattr(self.color_img, "shape"):
            info_dict["color_img"] = self.color_img
        else:
            info_dict["color_img"] = None

        if hasattr(self.depth_img, "shape"):
            info_dict["depth_img"] = self.depth_img
        else:
            info_dict["depth_img"] = None

        return info_dict
    

    def __str__(self):

        display_feedback = copy.deepcopy(self.to_dict())

        # Replace color_img and depth_img with shape info
        if hasattr(display_feedback.get("color_img"), "shape"):
            display_feedback["color_img"] = f"np.ndarray{display_feedback['color_img'].shape}"
        elif isinstance(display_feedback.get("color_img"), dict) and "shape" in display_feedback["color_img"]:
            display_feedback["color_img"] = f"np.ndarray{tuple(display_feedback['color_img']['shape'])}"

        if hasattr(display_feedback.get("depth_img"), "shape"):
            display_feedback["depth_img"] = f"np.ndarray{display_feedback['depth_img'].shape}"
        elif isinstance(display_feedback.get("depth_img"), dict) and "shape" in display_feedback["depth_img"]:
            display_feedback["depth_img"] = f"np.ndarray{tuple(display_feedback['depth_img']['shape'])}"

        return json.dumps(display_feedback, indent=2)