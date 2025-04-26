class Time:
    def __init__(self, sec: int = 0, nanosec: int = 0):
        self.sec = sec
        self.nanosec = nanosec

    def to_dict(self):
        return {
            "sec": self.sec,
            "nanosec": self.nanosec,
        }