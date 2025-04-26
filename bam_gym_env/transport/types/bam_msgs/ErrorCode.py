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