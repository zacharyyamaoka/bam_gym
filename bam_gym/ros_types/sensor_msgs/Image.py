# https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html

from bam_gym.ros_types.std_msgs.Header import Header
from typing import List

class Image:
    def __init__(self):
        self.header: Header = Header()

        self.height: int = 0
        self.width: int = 0

        self.encoding: str = ""

        self.is_bigendian: int = 0
        self.step: int = 0
        self.data: List[int] = []

    def to_dict(self):
        return {
            "header": self.header.to_dict(),

            "height": self.height,
            "width": self.width,

            "encoding": self.encoding,

            "is_bigendian": self.is_bigendian,
            "step": self.step,
            "data": self.data,
        }
    