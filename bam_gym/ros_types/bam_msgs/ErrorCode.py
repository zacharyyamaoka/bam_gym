from enum import IntEnum

class ErrorType(IntEnum):
    UNDEFINED = 0
    SUCCESS = 1
    FAILURE = 2
    UNABLE_TO_AQUIRE_SENSOR_DATA = 5

class ErrorCode:
    def __init__(self):
        self.value = ErrorType.UNDEFINED

    def to_dict(self):
        return {"value": int(self.value)}
    
    @classmethod
    def from_dict(cls, d: dict):
        obj = cls()
        raw_value = d.get("value", 0)
        return obj
    
    def name(self):
        """Return the name of the error (e.g., 'SUCCESS', 'FAILURE')."""
        return self.value.name

    def __str__(self):
        return self.value.name