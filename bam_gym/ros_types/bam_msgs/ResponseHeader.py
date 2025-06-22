from bam_gym.ros_types.bam_msgs import ErrorCode
from bam_gym.ros_types.builtin_interfaces import Time

class ResponseHeader:
    def __init__(self):
        self.process_duration: float = 0.0
        self.transport_duration: float = 0.0

        self.error_code = ErrorCode()
        self.error_msg = ""
        self.calibrated = False
        self.response_stamp = Time()

    def to_dict(self):
        return {
            "process_duration": self.process_duration,
            "transport_duration": self.transport_duration,
            "error_code": self.error_code.to_dict(),
            "error_msg": self.error_msg,
            "calibrated": self.calibrated,
            "response_stamp": self.response_stamp.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls()
        obj.process_duration = d.get("process_duration", 0.0)
        obj.transport_duration = d.get("transport_duration", 0.0)
        # Handle nested error_code field properly
        obj.error_code = ErrorCode.from_dict(d.get("error_code", {}))
        obj.error_msg = d.get("error_msg", "")
        obj.response_stamp = Time.from_dict(d.get("response_stamp", {}))

        return obj