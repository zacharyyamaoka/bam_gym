from typing import List
import numpy as np
import json

class GymResponse:
    def __init__(self):
        self.ns = ""
        self.observation: List[float] = []
        self.color_img = None
        self.depth_img = None
        self.reward: List[float] = []
        self.reward_action_index: List[int] = []
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
            "reward_action_index": self.reward_action_index,
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
        obj.reward_action_index = d.get("reward_action_index", [])
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

        if hasattr(self.color_img, "shape"):
            info_dict["color_img"] = self.color_img
        else:
            info_dict["color_img"] = None

        if hasattr(self.depth_img, "shape"):
            info_dict["depth_img"] = self.depth_img
        else:
            info_dict["depth_img"] = None

        return info_dict