from bam_gym.ros_types.builtin_interfaces import Time  # We'll define Time too if you don't have it yet

class Header:
    def __init__(self, stamp=None, frame_id: str = ""):
        self.stamp = stamp if stamp is not None else Time()
        self.frame_id = frame_id

    def to_dict(self):
        return {
            "stamp": self.stamp.to_dict(),
            "frame_id": self.frame_id,
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        stamp = Time.from_dict(d.get("stamp", {})) #empty dict otherwise
        frame_id = d.get("frame_id", "")
        return cls(stamp=stamp, frame_id=frame_id)