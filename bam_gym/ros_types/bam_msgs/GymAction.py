from enum import IntEnum
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
        self.rank: int = 0
        self.linked_actions: List[int] = []

    def to_dict(self):
        return {
            "action_type": self.action_type,
            "ns": self.ns,
            "prefix": self.prefix,
            "discrete_action": [int(a) for a in self.discrete_action],
            "contious_action": [float(a) for a in self.contious_action],
            "rank": int(self.rank),
            "linked_actions": self.linked_actions,
        }