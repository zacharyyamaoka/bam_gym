# https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Polygon.html
# its actually Point32 but has same fields as Point
from bam_gym.ros_types.geometry_msgs.Point import Point
from typing import List

class Polygon:
    def __init__(self, points=None):
        self.points: List[Point] = points if points is not None else []  # List of [x, y]

    def to_dict(self):
        return {
            "points": [p.to_dict() for p in self.points],
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        points_data = d.get("points", [])
        points = [Point.from_dict(p) for p in points_data]
        return cls(points=points)