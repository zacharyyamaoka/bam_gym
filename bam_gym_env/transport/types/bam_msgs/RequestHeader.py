from builtin_interfaces.Time import Time

class RequestHeader:
    def __init__(self):
        self.client_id = ""
        self.request_type: int = 0 
        self.force_if_busy = False
        self.stamp = Time()
        self.frame_id = ""
        self.expected_duration: float = 0.0
        self.blocking = False

    def to_dict(self):
        return {
            "client_id": self.client_id,
            "request_type": self.request_type,
            "force_if_busy": self.force_if_busy,
            "stamp": self.stamp.to_dict(),
            "frame_id": self.frame_id,
            "expected_duration": self.expected_duration,
            "blocking": self.blocking,
        }