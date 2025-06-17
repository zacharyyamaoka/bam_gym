import numpy as np

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return [qx, qy, qz, qw]

class Quaternion:
    def __init__(self, x = 0.0, y = 0.0, z = 0.0, w = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "w": self.w,
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        if "w" in d:
            return cls(x=d.get("x", 0.0), y=d.get("y", 0.0), z=d.get("z", 0.0), w=d.get("w", 1.0))
        else:
            qx, qy, qz, qw = get_quaternion_from_euler(d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0))
            return cls(x=qx, y=qy, z=qz, w=qw)
