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
            "duplicate_obs_ns": self.duplicate_obs_ns,
            "executed": self.executed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }
    

    @classmethod
    def from_dict(cls, d: dict): # from json dict of GymFeedback.msg
        obj = cls()
        obj.ns = d.get("ns", "")
        obj.observation = d.get("observation", [])
        obj.color_img = d.get("color_img", None)
        obj.depth_img = d.get("depth_img", None)
        obj.reward = d.get("reward", [])
        obj.duplicate_obs_ns = d.get("duplicate_obs_ns", False)
        obj.executed = d.get("executed", False)
        obj.terminated = d.get("terminated", False)
        obj.truncated = d.get("truncated", False)
        obj.info = d.get("info", "{}")
        return obj

    def obs_dict(self):
        obs_dict = {}
        if self.observation:
            obs_dict["obs"] = np.array(self.observation, dtype=np.float32)

        if isinstance(self.color_img, np.ndarray):
            obs_dict["color_img"] = self.color_img

        if isinstance(self.depth_img, np.ndarray):
            obs_dict["depth_img"] = self.depth_img

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

        info_dict['ns'] = self.ns
        info_dict['duplicate_obs_ns'] = self.duplicate_obs_ns
        info_dict['executed'] = self.executed

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