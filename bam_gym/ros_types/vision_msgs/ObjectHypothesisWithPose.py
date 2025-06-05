# https://docs.ros.org/en/rolling/p/vision_msgs/msg/ObjectHypothesisWithPose.html

from bam_gym.ros_types.geometry_msgs.PoseWithCovariance import PoseWithCovariance
from bam_gym.ros_types.vision_msgs.ObjectHypothesis import ObjectHypothesis

from typing import List

class ObjectHypothesisWithPose:
    def __init__(self):
        self.hypothesis: ObjectHypothesis = ObjectHypothesis()
        self.pose: PoseWithCovariance = PoseWithCovariance()

    def to_dict(self):
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "pose": self.pose.to_dict(),
        }
