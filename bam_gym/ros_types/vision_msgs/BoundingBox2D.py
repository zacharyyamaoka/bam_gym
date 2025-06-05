# https://docs.ros.org/en/rolling/p/vision_msgs/msg/BoundingBox2D.html
from bam_gym.ros_types.vision_msgs.Pose2D import Pose2D

class BoundingBox2D:
    def __init__(self):
        self.center: Pose2D = Pose2D()
        self.size_x: float = 0.0 
        self.size_y: float = 0.0 

    def to_dict(self):
        return {
            "center": self.center.to_dict(),
            "size_x": self.size_x,
            "size_y": self.size_y

        }
