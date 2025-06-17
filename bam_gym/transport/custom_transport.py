
# Implement another custom transport
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode, ErrorType, GymFeedback

class MockTransport():
    def __init__(self):
        pass

    def step(self, request: GymAPI_Request) -> GymAPI_Response:

     
        response = GymAPI_Response()
        response.header.error_code.value = ErrorType.SUCCESS
        response.header.error_msg = "dummy msg from mock transport"

        for action in request.action:
            f = GymFeedback()
            response.feedback.append(f)
       
        return response
