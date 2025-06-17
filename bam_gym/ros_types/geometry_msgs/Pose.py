from bam_gym.ros_types.geometry_msgs.Point import Point
from bam_gym.ros_types.geometry_msgs.Quaternion import Quaternion

class Pose:
    def __init__(self, position=None, orientation=None):

        if isinstance(position, Point):
            self.position = position
        elif isinstance(position, list) or isinstance(position, tuple):
            self.position = Point(*position)
        elif position is None:
            self.position = Point()
        else:
            assert False

        # TODO add support for passing in quaternion
        if isinstance(orientation, Quaternion):
            self.orientation = orientation
        elif isinstance(orientation, list) or isinstance(orientation, tuple):
            # Treat as Euler angles
            r_dict = {
                "x": orientation[0],
                "y": orientation[1],
                "z": orientation[2]
            }
            if len(orientation) == 4:
                r_dict['w'] = orientation[3]

            self.orientation = Quaternion.from_dict(r_dict)
        elif orientation is None:
            self.orientation = Quaternion()
        else:
            assert False

    def to_dict(self):
        return {
            "position": self.position.to_dict(),
            "orientation": self.orientation.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        position = Point.from_dict(d["position"])
        orientation = Quaternion.from_dict(d["orientation"])

        return cls(position=position, orientation=orientation)