from enum import IntEnum

class ErrorType(IntEnum):
    UNDEFINED = 0
    SUCCESS = 1
    FAILURE = 2
    UNABLE_TO_AQUIRE_SENSOR_DATA = 5
    REQUEST_FAILURE = 5

class ErrorCode:
    def __init__(self, value=ErrorType.UNDEFINED.value):
        self.value = value

    def to_dict(self):
        return {"value": int(self.value)}
    
    @classmethod
    def from_dict(cls, d: dict):
        obj = cls()
        obj.value = d.get("value")

        return obj
    
    def name(self):
        """Return the name of the error (e.g., 'SUCCESS', 'FAILURE')."""
        return ErrorType(self.value).name

    def __str__(self):
        return (ErrorType(self.value).name, self.value)