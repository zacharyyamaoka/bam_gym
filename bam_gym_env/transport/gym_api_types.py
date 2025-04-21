from enum import IntEnum
from typing import List, Optional, Dict, Any
import time
import json
import numpy as np

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


class ErrorCode:
    UNDEFINED = 0
    SUCCESS = 1
    FAILURE = 2
    # Add more codes as needed

    def __init__(self, value: int = SUCCESS):
        self.value = value

    def to_dict(self):
        return {"value": self.value}


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


class GymAPIRequest:
    def __init__(self,
                 header: RequestHeader = None,
                 seed: int = 0, # for ROS msg, int is required
                 env_name: str = "",
                 discrete_action: List[int] = None,
                 contious_action: List[float] = None):
        
        def ensure_list(x):
            if x is None:
                return []
            elif isinstance(x, (list, tuple)):
                return list(x)
            else:
                return [x]
            
        discrete_action = ensure_list(discrete_action)
        contious_action = ensure_list(contious_action)

        self.env_name = env_name
        self.seed = seed
        self.header = header or RequestHeader()
        self.discrete_action = discrete_action
        self.contious_action = contious_action 

    def to_dict(self):

        return {
            "header": self.header.to_dict(),
            "seed": self.seed,
            "env_name": self.env_name,
            "discrete_action": [int(x) for x in self.discrete_action],
            "contious_action": [float(x) for x in self.contious_action],
        }


class GymAPIResponse:
    def __init__(self, response: dict): # Always create from something, don't autopopulate
        self.header = ResponseHeader.from_dict(response.get("header", {}))
        self.observation = response.get("observation", [])
        self.color_img = response.get("color_img", None)
        self.depth_img = response.get("depth_img", None)
        self.reward = response.get("reward", 0.0)
        self.terminated = response.get("terminated", False)
        self.truncated = response.get("truncated", False)
        self.info = response.get("info", "{}")

    def to_step_tuple(self):
        return (
            np.array(self.observation, dtype=np.float32),
            self.reward,
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

        info_dict["color_img"] = self.color_img
        info_dict["depth_img"] = self.depth_img
        info_dict["header"] = self.header.to_dict() 

        return info_dict


