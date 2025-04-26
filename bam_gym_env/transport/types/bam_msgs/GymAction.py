from enum import IntEnum
from geometry_msgs.PoseStamped import PoseStamped
from bam_msgs.WaypointParams import WaypointParams
from typing import List

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
        self.discrete_action: List[int] = []
        self.contious_action: List[float] = []
        self.pose_action: List[PoseStamped] = []
        self.parameters: List[WaypointParams] = []
        self.rank: int = 0
        self.linked_actions: List[int] = []

    def to_dict(self):
        return {
            "action_type": int(self.action_type),
            "ns": self.ns,
            "prefix": self.prefix,
            "discrete_action": self.discrete_action,
            "contious_action": self.contious_action,
            "pose_action": [p.to_dict() for p in self.pose_action],
            "parameters": [p.to_dict() for p in self.parameters],
            "rank": self.rank,
            "linked_actions": self.linked_actions,
        }