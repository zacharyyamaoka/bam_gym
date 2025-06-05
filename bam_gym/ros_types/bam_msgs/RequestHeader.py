from bam_gym.ros_types.builtin_interfaces import Time

class RequestHeader:
    def __init__(self):
        self.client_id = ""
        self.request_type: int = 0 

        self.force_busy = False
        self.force_uncalibrated = False
        self.force_priority = False

        self.stamp = Time()
        self.frame_id = ""
        self.expected_duration: float = 0.0
        self.blocking = False

        self.priority = 0
        self.clear_priority = False

    def to_dict(self):
        return {
            "client_id": self.client_id,
            "request_type": self.request_type,

            "force_busy": self.force_busy,
            "force_uncalibrated": self.force_uncalibrated,
            "force_priority": self.force_priority,

            "stamp": self.stamp.to_dict(),
            "frame_id": self.frame_id,
            "expected_duration": self.expected_duration,
            "blocking": self.blocking,

            "priority": self.priority,
            "clear_priority": self.clear_priority,

        }