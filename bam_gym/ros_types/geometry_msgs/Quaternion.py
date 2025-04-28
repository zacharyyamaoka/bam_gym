class Quaternion:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "w": self.w,
        }