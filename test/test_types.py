#!/usr/bin/env python3

import json
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import GymAction


action = GymAction()
print("\n\nGymAction")
print(json.dumps(action.to_dict(), indent=2))

request = GymAPI_Request()
print("\n\nGymAPI_Request")
print(json.dumps(request.to_dict(), indent=2))

response = GymAPI_Response()

print("\n\nGymAPI_Response")
print(json.dumps(response.to_dict(), indent=2))
