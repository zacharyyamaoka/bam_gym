
# Implement another custom transport
from ros_py_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from ros_py_types.bam_msgs import ErrorCode, ErrorType, GymFeedback
from bam_gym.transport.generic_transport import GenericTransport

class LocalTransport(GenericTransport):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, request: GymAPI_Request) -> GymAPI_Response:
        response = GymAPI_Response()
        response.header.error_code.value = ErrorType.SUCCESS
        response.header.error_msg = "dummy msg from mock transport"

        for action in request.action:
            f = GymFeedback()
            response.feedback.append(f)
       
        return response
