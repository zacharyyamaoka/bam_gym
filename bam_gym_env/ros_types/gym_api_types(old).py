from enum import IntEnum
from typing import List, Optional, Dict, Any
import time
import json
import numpy as np


def ensure_list(var):
    if var is None:
        return []
    elif isinstance(var, (list, tuple)):
        return list(var)
    else:
        return [var]
        
#region - REQUEST

class RequestType(IntEnum):
    NONE = 0
    STEP = 1
    RESET = 2
    CLOSE = 3

class TimeStamp:
    def __init__(self, secs: int = None, nsecs: int = None):

        now = time.time()
        if secs == None:
            secs = int(now)
        if nsecs == None:
            nsecs = int((now - secs) * 1e9)   

        self.secs = secs
        self.nsecs = nsecs

    def to_dict(self):
        return {"sec": self.secs, "nanosec": self.nsecs}

class RequestHeader:
    def __init__(self,
                 client_id: str = "",
                 request_type: RequestType = None,
                 force_if_busy: bool = False,
                 frame_id: str = "",
                 expected_duration: float = 0.0,
                 stamp: TimeStamp = None):
        
        self.client_id = client_id
        self.request_type = request_type or RequestType.NONE
        self.force_if_busy = force_if_busy
        self.stamp = stamp or TimeStamp()
        self.frame_id = frame_id
        self.expected_duration = expected_duration

    def to_dict(self):

        return {
            "client_id": self.client_id,
            "request_type": int(self.request_type),
            "force_if_busy": self.force_if_busy,
            "stamp": self.stamp.to_dict(),
            "frame_id": self.frame_id,
            "expected_duration": self.expected_duration,
        }

class ActionType(IntEnum):
    DEFAULT = 1
    PICK_AND_LIFT = 2
    PICK_AND_PLACE = 3
    OPEN = 4
    FOLD_SHIRT = 5

class WaypointParams:
    def __init__(self, params: dict = None):
        self.params = params or {}

    def to_dict(self):
        return self.params

class PoseStamped:
    def __init__(self, frame_id: str = "", position: List[float] = None, orientation: List[float] = None):
        self.frame_id = frame_id
        self.position = position or [0.0, 0.0, 0.0]  # x, y, z
        self.orientation = orientation or [0.0, 0.0, 0.0, 1.0]  # x, y, z, w (quaternion)

    def to_dict(self):
        return {
            "header": {"frame_id": self.frame_id},
            "pose": {
                "position": {
                    "x": self.position[0],
                    "y": self.position[1],
                    "z": self.position[2],
                },
                "orientation": {
                    "x": self.orientation[0],
                    "y": self.orientation[1],
                    "z": self.orientation[2],
                    "w": self.orientation[3],
                }
            }
        }
    
class GymAction:
    def __init__(self,
                 action_type: ActionType = ActionType.DEFAULT,
                 ns: str = "",
                 prefix: str = "",
                 discrete_action: List[int] = None,
                 contious_action: List[float] = None,
                 pose_action: List[PoseStamped] = None,
                 parameters: List[WaypointParams] = None,
                 rank: int = 0,
                 linked_actions: List[int] = None):
        
        self.action_type = action_type
        self.ns = ns
        self.prefix = prefix
        self.discrete_action = ensure_list(discrete_action)
        self.contious_action = ensure_list(contious_action)
        self.pose_action = ensure_list(pose_action) 
        self.parameters = ensure_list(parameters) 
        self.rank = rank
        self.linked_actions = ensure_list(linked_actions)
        
    def to_dict(self):
        return {
            "action_type": int(self.action_type),
            "ns": self.ns,
            "prefix": self.prefix,
            "discrete_action": self.discrete_action,
            "contious_action": self.contious_action,
            "pose_action": self.pose_action, # assumed to already be dictionaries
            "parameters": [param.to_dict() for param in self.parameters],
            "rank": self.rank,
            "linked_actions": self.linked_actions,
        }
       
class GymAPIRequest:
    def __init__(self,
                 header: RequestHeader = None,
                 seed: int = 0, # for ROS msg, int is required
                 env_name: str = "",
                 action: List[GymAction] = None):
        
        self.header = header or RequestHeader()
        self.seed = seed
        self.env_name = env_name
        self.action = action or []

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "seed": self.seed,
            "env_name": self.env_name,
            "action": [a.to_dict() for a in self.action],
        }

#endregion - REQUEST


#region - RESPONSE

class ErrorCode:
    UNDEFINED = 0
    SUCCESS = 1
    FAILURE = 2
    # Add more codes as needed

    def __init__(self, value: int = SUCCESS):
        self.value = value

    def to_dict(self):
        return {"value": self.value}
    
    @classmethod
    def name(cls, value: int) -> str:
        for attr in dir(cls):
            if not attr.startswith("_") and isinstance(getattr(cls, attr), int):
                if getattr(cls, attr) == value:
                    return f"{attr} ({value})"
        return f"UNKNOWN ({value})"
    
class ResponseHeader:
    def __init__(self,
                 duration: float = 0.0,
                 error_code: ErrorCode = None,
                 error_msg: str = ""):
        
        self.duration = duration
        self.error_code = error_code or ErrorCode()
        self.error_msg = error_msg

    # Clean way to instate object from a dict
    # Use like header = ResponseHeader.from_dict()
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            duration=d.get("duration", 0.0),
            error_code=ErrorCode(d.get("error_code", {}).get("value", ErrorCode.FAILURE)),
            error_msg=d.get("error_msg", "")
        )
    
    def to_dict(self):
        return {
            "duration": self.duration,
            "error_code": self.error_code.to_dict(),
            "error_msg": self.error_msg,
        }

class GymFeedback:
    def __init__(self,
                 observation: List[float] = None,
                 color_img: Any = None,
                 depth_img: Any = None,
                 reward: List[float] = None,
                 reward_action_index: List[int] = None,
                 terminated: bool = False,
                 truncated: bool = False,
                 info: str = ""):
        
        self.observation = observation or []
        self.color_img = color_img
        self.depth_img = depth_img
        self.reward = reward or []
        self.reward_action_index = reward_action_index or []
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            observation=d.get("observation", []),
            color_img=d.get("color_img", None),
            depth_img=d.get("depth_img", None),
            reward=d.get("reward", []),
            reward_action_index=d.get("reward_action_index", []),
            terminated=d.get("terminated", False),
            truncated=d.get("truncated", False),
            info=d.get("info", "{}"),
        )

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

class GymAPIResponse:
    def __init__(self, response: dict):
        self.header = ResponseHeader.from_dict(response.get("header", {}))
        self.response = GymFeedback.from_dict(response.get("response", {}))

    def to_step_tuple(self):
        return self.response.to_step_tuple()

    def to_reset_tuple(self):
        return self.response.to_reset_tuple()
    
#endregion - RESPONSE
