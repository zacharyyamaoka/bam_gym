from enum import IntEnum
from typing import List, Optional, Dict, Any
import numpy as np

from bam_msgs.RequestHeader import RequestHeader
from bam_msgs.ResponseHeader import ResponseHeader
from bam_msgs.GymAction import GymAction
from bam_msgs.GymResponse import GymResponse

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
        self.response = GymResponse()

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "response": self.response.to_dict(),
        }
    
    def to_step_tuple(self):
        return self.response.to_step_tuple()

    def to_reset_tuple(self):
        return self.response.to_reset_tuple()

