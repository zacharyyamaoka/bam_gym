class Time:
    def __init__(self, sec: int = 0, nanosec: int = 0):
        self.sec = sec
        self.nanosec = nanosec

    def to_dict(self):
        return {
            "sec": self.sec,
            "nanosec": self.nanosec,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(sec=d.get("sec", 0), nanosec=d.get("nanosec", 0))