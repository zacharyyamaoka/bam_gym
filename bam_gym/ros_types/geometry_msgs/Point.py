# https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Point.html

class Point:
    def __init__(self, x = 0.0, y = 0.0, z= 0.0):
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            z=d.get("z", 0.0),
        )