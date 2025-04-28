from bam_gym.ros_types.std_msgs.Header import Header
from bam_gym.ros_types.geometry_msgs.Pose import Pose

class PoseStamped:
    def __init__(self, header=None, pose=None):
        self.header = header if header is not None else Header()
        self.pose = pose if pose is not None else Pose()

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "pose": self.pose.to_dict(),
        }