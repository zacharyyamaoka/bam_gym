import roslibpy
import time
import base64
import numpy as np
import cv2
from ros_py_types.bam_srv import GymAPI_Request, GymAPI_Response, RequestType
from ros_py_types.bam_msgs import ErrorCode, ErrorType
import json

# https://roslibpy.readthedocs.io/en/latest/examples.html

from ros_py_types.bam_msgs import ErrorCode, ErrorType, GymFeedback
from bam_gym.transport.generic_transport import GenericTransport


class RoslibpyTransport(GenericTransport):
    def __init__(self, namespace="", node_name="gym_env", host_ip='localhost', port=9090, timeout_sec=5, **kwargs):
        """ 
        namespace: Each robot rack has a unique namespace (bam_GPU, bam_001, etc.)
        node_name: by default is set to 'gym_env'
        """
        super().__init__(**kwargs)

        namespace = namespace.strip('/')
        if namespace:
            namespace = '/' + namespace  # Add leading slash if not empty
        node_name = node_name.strip('/')
        service_topic = namespace + "/" + node_name + "/gym_api_request"
        service_type = 'bam_msgs/GymAPI'

        self.client = roslibpy.Ros(host=host_ip, port=port)

        print(f'Trying to connect to ROS bridge at {host_ip}:{port}...')
        self.client.run(timeout=3)

        self.timeout_sec = timeout_sec
        # Wait until connected
        while not self.client.is_connected:
            print('Waiting for ROS bridge connection...')
            time.sleep(1)

        print('Connection Succesful.')

        self.service = roslibpy.Service(self.client, service_topic, service_type)

    def error_response(self, msg=''):
        response = GymAPI_Response()
        response.header.error_code = ErrorCode(ErrorType.FAILURE.value)
        response.header.error_msg = f"{msg}"
        return response
    
    def step(self, request: GymAPI_Request) -> GymAPI_Response:
        
        # Convert GymAPI_Request -> Roslibpy Request (excepts a json)
        ros_request = roslibpy.ServiceRequest(request.to_dict())

        try:
            ros_response = self.service.call(ros_request, timeout=self.timeout_sec)

            if ros_response == None:
                print("[roslibpy_transport] service call return with no response")
                return self.error_response("[roslibpy_transport] service call return with no response")

        except Exception as e:
            print(f"[roslibpy_transport][ERROR] {e}")
            print("[roslibpy_transport] Make sure environment is running on server")

            return self.error_response(e)

        # Roslibpy Response to Convert GymAPI_Response 

        # TODO POPULATE RESPONSE HERE
        # Decompress image, do this in here so gym layer doesn't need to worry about it
        # Output either the numpy array or none
        response = GymAPI_Response.from_dict(ros_response)
        self.decompress_imgs(response) # update in place

        return response


        
    def decompress_imgs(self, response: GymAPI_Response):
        # See: https://roslibpy.readthedocs.io/en/latest/examples/05_subscribe_to_images.html
        # https://answers.ros.org/question/333329/
        # Updates images in place

        for feedback in response.feedback:
            # Decompress all color images in the list
            for img in feedback.color_img:
                if isinstance(img.data, str) and len(img.data) > 0:
                    img.data = self.img_data_decode(img.data, img_type="color")
            # Decompress all depth images in the list
            for img in feedback.depth_img:
                if isinstance(img.data, str) and len(img.data) > 0:
                    img.data = self.img_data_decode(img.data, img_type="depth")
            # Decompress mask images in all segments (Segment2DArray)
            for seg_array in feedback.segments:
                for seg in seg_array.segments:
                    if isinstance(seg.mask.data, str) and len(seg.mask.data) > 0:
                        seg.mask.data = self.img_data_decode(seg.mask.data, img_type="mask")


    def img_data_decode(self, img_data: str, img_type="color"):
        """Helper to decompress a single image string. img_type: 'color', 'depth', or 'mask'."""
        try:
            decoded_bytes = base64.b64decode(img_data.encode('ascii'))
            np_arr = np.frombuffer(decoded_bytes, dtype=np.uint8)
            if img_type == "color":
                img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif img_type == "depth":
                img_cv = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED) 
                if isinstance(img_cv, np.ndarray):
                    img_cv/= 1000.0
            elif img_type == "mask":
                img_cv = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            else:
                raise ValueError(f"Unknown img_type: {img_type}")
            return img_cv
        except Exception as e:
            print(f"[ERROR decompressing {img_type}]: {e}")
            return None
        
    def close(self):
        self.client.terminate()

def main():

    transport = RoslibpyTransport('bam_GPU')

    request = GymAPI_Request()
    request.env_name = 'cart_pole'
    request.header.request_type = RequestType.RESET
    response = transport.step(request)
    print(response)



if __name__ == '__main__':
    main()
