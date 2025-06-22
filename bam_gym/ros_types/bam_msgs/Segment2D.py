from bam_gym.ros_types.std_msgs.Header import Header
from bam_gym.ros_types.vision_msgs.ObjectHypothesisWithPose import ObjectHypothesisWithPose
from bam_gym.ros_types.vision_msgs.BoundingBox2D import BoundingBox2D
from bam_gym.ros_types.geometry_msgs.Polygon import Polygon
from bam_gym.ros_types.sensor_msgs.Image import Image

class Segment2D:
    def __init__(self):
        self.header = Header()
        self.results = []
        self.bbox = BoundingBox2D()
        self.id = ""
        self.polygon = []
        self.mask = Image()
        self.bbox_mask = False
        self.img_height = 0
        self.img_width = 0

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "bbox": self.bbox.to_dict(),
            "id": self.id,
            "polygon": [p.to_dict() for p in self.polygon],
            "mask": self.mask.to_dict(),
            "bbox_mask": self.bbox_mask,
            "img_height": self.img_height,
            "img_width": self.img_width,
        }

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls()
        obj.header = Header.from_dict(d.get("header", {}))
        obj.results = [ObjectHypothesisWithPose.from_dict(r) for r in d.get("results", [])]
        obj.bbox = BoundingBox2D.from_dict(d.get("bbox", {}))
        obj.id = d.get("id", "")
        obj.polygon = [Polygon.from_dict(p) for p in d.get("polygon", [])]
        obj.mask = Image.from_dict(d.get("mask", {}))
        obj.bbox_mask = d.get("bbox_mask", False)
        obj.img_height = d.get("img_height", 0)
        obj.img_width = d.get("img_width", 0)
        return obj
