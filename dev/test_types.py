#!/usr/bin/env python3

import json
from bam_gym.ros_types.bam_srv import GymAPIRequest, GymAPIResponse, RequestType
from bam_gym.ros_types.bam_msgs import GymAction


action = GymAction()
print("\n\nGymAction")
print(json.dumps(action.to_dict(), indent=2))

request = GymAPIRequest()
print("\n\nGymAPIRequest")
print(json.dumps(request.to_dict(), indent=2))

response = GymAPIResponse()

print("\n\nGymAPIResponse")
print(json.dumps(response.to_dict(), indent=2))
