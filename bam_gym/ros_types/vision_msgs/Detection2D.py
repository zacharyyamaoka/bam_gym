# https://docs.ros.org/en/rolling/p/vision_msgs/msg/Detection2D.html

from bam_gym.ros_types.std_msgs.Header import Header
from bam_gym.ros_types.sensor_msgs.Image import Image
from bam_gym.ros_types.vision_msgs.ObjectHypothesisWithPose import ObjectHypothesisWithPose
from bam_gym.ros_types.vision_msgs.BoundingBox2D import BoundingBox2D

from typing import List

class Detection2D:
    def __init__(self):
        self.header: Header = Header()
        self.results: List[ObjectHypothesisWithPose] = []
        self.bbox: BoundingBox2D = BoundingBox2D()
        self.id: str = ""

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "bbox": self.bbox.to_dict(),
            "id": self.id,

        }

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls()
        obj.header = Header.from_dict(d.get("header", {}))
        obj.results = [ObjectHypothesisWithPose.from_dict(r) for r in d.get("results", [])]
        obj.bbox = BoundingBox2D.from_dict(d.get("bbox", {}))
        obj.id = d.get("id", "")
        return obj