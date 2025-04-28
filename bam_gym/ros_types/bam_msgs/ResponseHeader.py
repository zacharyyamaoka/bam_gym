from bam_gym.ros_types.bam_msgs import ErrorCode

class ResponseHeader:
    def __init__(self):
        self.duration: float = 0.0
        self.error_code = ErrorCode()
        self.error_msg = ""

    def to_dict(self):
        return {
            "duration": self.duration,
            "error_code": self.error_code.to_dict(),
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls()
        obj.duration = d.get("duration", 0.0)

        # Handle nested error_code field properly
        error_code_dict = d.get("error_code", {})
        obj.error_code = ErrorCode.from_dict(error_code_dict)
        obj.error_msg = d.get("error_msg", "")
        return obj