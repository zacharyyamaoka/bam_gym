# https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Point.html

class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }