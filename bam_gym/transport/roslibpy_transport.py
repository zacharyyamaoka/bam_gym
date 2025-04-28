import roslibpy
import time
import base64
import numpy as np
import cv2
from bam_gym.ros_types.bam_srv import GymAPIRequest, GymAPIResponse, RequestType
from bam_gym.ros_types.bam_msgs import ErrorCode

# https://roslibpy.readthedocs.io/en/latest/examples.html


class RoslibpyTransport():
    def __init__(self, namespace="", node_name="gym_env", host_ip='localhost', port=9090, timeout_sec=5):
        """ 
        namespace: Each robot rack has a unique namespace (bam_GPU, bam_001, etc.)
        node_name: by default is set to 'gym_env'
        """
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

    def step(self, request: GymAPIRequest) -> GymAPIResponse:
        
        # try:
        print(request.to_dict())
        ros_request = roslibpy.ServiceRequest(request.to_dict())

        ros_response = self.service.call(ros_request, timeout=self.timeout_sec)

        print("ROS RAW RESPONSE")
        # print(ros_response)

        # TODO POPULATE RESPONSE HERE
        # Decompress image, do this in here so gym layer doesn't need to worry about it
        # self.decompress_imgs(ros_response)
        response = GymAPIResponse.from_dict(ros_response)
        return response

        # except Exception as e:
        #     print(f"[ERROR] {e}")
        #     print("Make sure environment is running on server")
        #     response = GymAPIResponse()
        #     # response.header.error_code = ErrorCode(ErrorCode.FAILURE)
        #     # response.header.error_msg = f"{e}"
        #     return response
        
    def decompress_imgs(self, response):
        # See: https://roslibpy.readthedocs.io/en/latest/examples/05_subscribe_to_images.html
        # https://answers.ros.org/question/333329/
        feedback_list = response.get('feedback', [])

        for feedback in feedback_list:
            # Decompress color_img if exists
            if isinstance(feedback.get('color_img'), dict):
                feedback['color_img'] = self._decompress_img(feedback['color_img'])

            # Decompress depth_img if exists (optional)
            if isinstance(feedback.get('depth_img'), dict):
                feedback['depth_img'] = self._decompress_img(feedback['depth_img'], color=False)

        return response

    def _decompress_img(self, img_data, color=True):
        """Helper to decompress a single CompressedImage dict."""
        if img_data is None or 'data' not in img_data:
            return None

        if len(img_data['data']) == 0:
            return None

        try:
            decoded_bytes = base64.b64decode(img_data['data'].encode('ascii'))
            np_arr = np.frombuffer(decoded_bytes, dtype=np.uint8)
            if color:
                img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                img_cv = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            return img_cv
        except Exception as e:
            print(f"[ERROR decompressing image]: {e}")
            return None
        
    def close(self):
        self.client.terminate()

def main():

    transport = RoslibpyTransport('bam_GPU')

    request = GymAPIRequest()
    request.env_name = 'cart_pole'
    request.header.request_type = RequestType.RESET
    response = transport.step(request)
    print(response)



if __name__ == '__main__':
    main()
        