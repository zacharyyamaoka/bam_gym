# https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/RegionOfInterest.html


class RegionOfInterest:
    def __init__(self):
        self.x_offset: int = 0  # Leftmost pixel
        self.y_offset: int = 0  # Topmost pixel
        self.height: int = 0
        self.width: int = 0
        self.do_rectify: bool = False

    def to_dict(self):
        return {
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "height": self.height,
            "width": self.width,
            "do_rectify": self.do_rectify,
        }