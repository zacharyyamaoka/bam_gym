from bam_gym.ros_types.std_msgs.Header import Header
from bam_gym.ros_types.bam_msgs.Segment2D import Segment2D

from typing import List

class Segment2DArray:
    def __init__(self):
        self.header = Header()
        self.segments: List[Segment2D] = []

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls()
        obj.header = Header.from_dict(d.get("header", {}))
        obj.segments = [Segment2D.from_dict(s) for s in d.get("segments", [])]
        return obj
