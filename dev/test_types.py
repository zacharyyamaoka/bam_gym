#!/usr/bin/env python3

import json
from bam_gym.ros_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from bam_gym.ros_types.bam_msgs import GymAction, GymFeedback, WaypointParams
from bam_gym.ros_types.geometry_msgs.Point import Point
from bam_gym.ros_types.geometry_msgs.PoseStamped import PoseStamped
from bam_gym.ros_types.geometry_msgs.Polygon import Polygon
from bam_gym.ros_types.vision_msgs.Detection2D import Detection2D
from bam_gym.ros_types.vision_msgs.ObjectHypothesisWithPose import ObjectHypothesisWithPose

action = GymAction()
action.pose_action = [PoseStamped()]
action.pose_param = [WaypointParams()]
# print("\n\nGymAction")
# print(json.dumps(action.to_dict(), indent=2))

feedback = GymFeedback()
detection = Detection2D()
detection.results = [ObjectHypothesisWithPose()]
feedback.detections = [detection]
mask = Polygon()
mask.points = [Point()]
feedback.masks = [mask]

# print("\n\nfeedback")
# print(json.dumps(action.to_dict(), indent=2))

request = GymAPI_Request()
request.action = [action]
# print("\n\nGymAPI_Request")
# print(json.dumps(request.to_dict(), indent=2))

response = GymAPI_Response()
response.feedback = [feedback]
# print("\n\nGymAPI_Response")
# print(json.dumps(response.to_dict(), indent=2))

src = GymAPI_Request()
for field in vars(src):
    print(field)
    value = getattr(src, field)
    print(value)


from typing import Any

def is_basic(value):
    if isinstance(value, (int, float, str)):
        return True
    if isinstance(value, list) and all(isinstance(v, (int, float, str)) for v in value):
        return True
    return False


def copy_fields(src: Any, dst: Any, exclude_fields: set = set(), depth=0, verbose=True):
    depth += 1
    def print_depth(msg):
        if verbose:
            print("   "*(depth-1) + msg)

    for field in vars(src):
        if field in exclude_fields:
            continue
        value = getattr(src, field)
        dst_value = getattr(dst, field)
        print_depth(f"src {field} : {value}")

        if is_basic(value):
            setattr(dst, field, value)
            print_depth(f"set dst {dst}.{field} = {value} ({type(value).__name__})")

        elif isinstance(value, list):
            # Handle list of objects (e.g., list of messages)
            copied_list = []
            for item in value:
                if is_basic(item):
                    copied_list.append(item)
                else:
                    assert len(dst_value) > 0
                    dst_item = type(dst_value[0])() # You must prepopulate list so the value can be accessed
                    copy_fields(item, dst_item, exclude_fields, depth)
                    copied_list.append(dst_item)

            setattr(dst, field, copied_list) # this will override the first random value
            print_depth(f"set dst {dst}.{field} = {copied_list}")

        elif hasattr(value, '__dict__'):
            # Nested message/object
            copy_fields(value, dst_value, exclude_fields, depth)
            setattr(dst, field, dst_value)
            print_depth(f"set {dst}.{field} = {dst_value} ({type(value).__name__})")
        else:
            raise ValueError(f"Unhandled field type for field '{field}'")
        

src = PoseStamped()
src.header.frame_id = "test"
src.header.stamp.nanosec = 1
src.header.stamp.sec = 1

src.pose.orientation.x = 1
src.pose.orientation.y = 1
src.pose.orientation.z = 1
src.pose.orientation.w = 1

src.pose.position.x = 1
src.pose.position.y = 1
src.pose.position.z = 1
dst = PoseStamped()
print(dst.to_dict())
copy_fields(src, dst)
print(PoseStamped().to_dict())
print(dst.to_dict())

src = PoseStamped()
src.header.frame_id = "test"
src.header.stamp.nanosec = 1
src.header.stamp.sec = 1

src.pose.orientation.x = 1
src.pose.orientation.y = 1
src.pose.orientation.z = 1
src.pose.orientation.w = 1

src.pose.position.x = 1
src.pose.position.y = 1
src.pose.position.z = 1
dst = PoseStamped()
print(dst.to_dict())
copy_fields(src, dst)
print(PoseStamped().to_dict())
print(dst.to_dict())
