# https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CompressedImage.html

from bam_gym.ros_types.std_msgs.Header import Header
from typing import List

class CompressedImage:
    def __init__(self):
        self.header: Header = Header()
        self.format: str = "None"
        self.data: List[int] = []

    def to_dict(self):
        return {
            "header": self.header.to_dict(),
            "format": self.format,
            "data": self.data,
        }
    