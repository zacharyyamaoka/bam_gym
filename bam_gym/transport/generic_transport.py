
# Implement another custom transport
from ros_py_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from ros_py_types.bam_msgs import ErrorCode, ErrorType, GymFeedback

class GenericTransport():
    def __init__(self, **kwargs):
        self.mock_response: GymAPI_Response | None = None

    def set_mock_response(self, response: GymAPI_Response):
        self.mock_response = response

    def step(self, request: GymAPI_Request) -> GymAPI_Response:
        response = GymAPI_Response()

        return response
