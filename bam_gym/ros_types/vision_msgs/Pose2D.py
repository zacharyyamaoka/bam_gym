# https://docs.ros.org/en/rolling/p/vision_msgs/msg/Pose2D.html
from bam_gym.ros_types.vision_msgs.Point2D import Point2D

class Pose2D:
    def __init__(self):
        self.position: Point2D = Point2D()
        self.theta: float = 0.0

    def to_dict(self):
        return {
            "position": self.position.to_dict(),
            "theta": self.theta
        }