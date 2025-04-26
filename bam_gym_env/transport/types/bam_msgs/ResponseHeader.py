from bam_msgs.ErrorCode import ErrorCode

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
