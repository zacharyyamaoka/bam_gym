from enum import IntEnum
from typing import List
from bam_gym.ros_types.geometry_msgs.PoseStamped import PoseStamped
from bam_gym.ros_types.bam_msgs.WaypointParams import WaypointParams

class ActionType(IntEnum):
    DEFAULT = 1
    PICK_AND_LIFT = 2
    PICK_AND_PLACE = 3
    OPEN = 4
    FOLD_SHIRT = 5

class GymAction:
    def __init__(self):
        self.action_type = ActionType.DEFAULT
        self.ns = ""
        self.prefix = ""

        self.discrete_names: List[str] = []
        self.discrete_action: List[int] = []

        self.continuous_names: List[str] = []
        self.continuous_action: List[float] = []

        self.pose_names: List[str] = []
        self.pose_action: List[PoseStamped] = []
        self.pose_param: List[WaypointParams] = []

        self.target_class: List[int] = []
        self.rank: int = 0

    def to_dict(self):
        return {
            "action_type": self.action_type,
            "ns": self.ns,
            "prefix": self.prefix,

            "discrete_names": self.discrete_names,
            "discrete_action": [int(a) for a in self.discrete_action],

            "continuous_names": self.continuous_names,
            "continuous_action": [float(a) for a in self.continuous_action],

            "pose_names": self.pose_names,
            "pose_action": [a.to_dict() for a in self.pose_action],
            "pose_param": [p.to_dict() for p in self.pose_param],

            "target_class": self.target_class,
            "rank": int(self.rank),
        }