from geometry_msgs.Point import Point
from geometry_msgs.Quaternion import Quaternion

class Pose:
    def __init__(self, position=None, orientation=None):
        self.position = position if position is not None else Point()
        self.orientation = orientation if orientation is not None else Quaternion()

    def to_dict(self):
        return {
            "position": self.position.to_dict(),
            "orientation": self.orientation.to_dict(),
        }