# https://docs.ros.org/en/rolling/p/vision_msgs/msg/ObjectHypothesis.html

class ObjectHypothesis:
    def __init__(self):
        self.class_id: str = ""
        self.score: float = 0.0

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "score": self.score,
        }
