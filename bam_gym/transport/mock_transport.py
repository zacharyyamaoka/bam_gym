
# Implement another custom transport
from ros_py_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from ros_py_types.bam_msgs import ErrorCode, ErrorType, GymFeedback
from bam_gym.transport.generic_transport import GenericTransport


class MockTransport(GenericTransport):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def step(self, request: GymAPI_Request) -> GymAPI_Response:

        if self.mock_response: return self.mock_response
        response = GymAPI_Response()
        response.header.error_code.value = ErrorType.FAILURE # make it failure beacuse obs is empty...
        response.header.error_msg = "MockTransport: No server running, so no obs to return."

        for action in request.action:
            f = GymFeedback()
            response.feedback.append(f)
       
        return response
