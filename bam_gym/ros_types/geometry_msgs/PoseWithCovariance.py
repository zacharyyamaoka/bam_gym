# https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseWithCovariance.html
from bam_gym.ros_types.geometry_msgs.Pose import Pose
from typing import List

class PoseWithCovariance:
    def __init__(self):
        self.pose: Pose = Pose()
        self.covariance: List[float] = [0.0]*36 

    def to_dict(self):
        return {
            "pose": self.pose.to_dict(),
            "covariance": self.covariance,
        }