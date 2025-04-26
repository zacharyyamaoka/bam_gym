from builtin_interfaces.Time import Time  # We'll define Time too if you don't have it yet

class Header:
    def __init__(self, stamp=None, frame_id: str = ""):
        self.stamp = stamp if stamp is not None else Time()
        self.frame_id = frame_id

    def to_dict(self):
        return {
            "stamp": self.stamp.to_dict(),
            "frame_id": self.frame_id,
        }