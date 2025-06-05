# https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
# https://docs.ros2.org/foxy/api/sensor_msgs/msg/CameraInfo.html Its updated in ROS2

from bam_gym.ros_types.std_msgs.Header import Header
from bam_gym.ros_types.sensor_msgs.RegionOfInterest import RegionOfInterest

from typing import List
class CameraInfo:
    def __init__(self):
        # Image acquisition info
        self.header: Header = Header()
        self.height: int = 0
        self.width: int = 0

        # Calibration parameters
        self.distortion_model: str = ""
        self.d: List[float] = []           # Distortion coefficients
        self.k: List[float] = [0.0] * 9     # Intrinsic matrix
        self.r: List[float] = [0.0] * 9     # Rectification matrix
        self.p: List[float] = [0.0] * 12    # Projection matrix

        # Operational parameters
        self.binning_x: int = 0
        self.binning_y: int = 0
        self.roi: RegionOfInterest = RegionOfInterest()

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "height": self.height,
            "width": self.width,

            "distortion_model": self.distortion_model,
            "d": self.d,
            "k": self.k,
            "r": self.r,
            "p": self.p,

            "binning_x": self.binning_x,
            "binning_y": self.binning_y,
            "roi": self.roi.to_dict(),
        }