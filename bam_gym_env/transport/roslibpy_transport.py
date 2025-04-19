import roslibpy
import time
import base64
import numpy as np
import cv2
from bam_gym_env.transport import GymAPIRequest, GymAPIResponse, RequestType, ErrorCode

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

        self.client = roslibpy.Ros(host=host_ip, port=port)
        self.client.run()

        self.timeout_sec = timeout_sec

        print(f'Trying to connect to ROS bridge at {host_ip}:{port}...')
        # Wait until connected
        while not self.client.is_connected:
            print('Waiting for ROS bridge connection...')
            time.sleep(1)

        print('Connection Succesful.')

        self.service = roslibpy.Service(self.client, namespace + "/" + node_name + "/gym_api_request", 'bam_msgs/GymAPI')

    def step(self, request: GymAPIRequest) -> GymAPIResponse:

        # try:
        ros_request = roslibpy.ServiceRequest(request.to_dict())

        ros_response = self.service.call(ros_request, timeout=self.timeout_sec)

        # Decompress image, do this in here so gym layer doesn't need to worry about it
        self.decompress_img(ros_response)

        response = GymAPIResponse(ros_response)
        return response

        # except Exception as e:
        #     print(f"[ERROR] {e}")
        #     response = GymAPIResponse(dict())
        #     response.header.error_code = ErrorCode.FAILURE
        #     response.header.error_msg = f"{e}"
        #     return response
        
    def decompress_img(self, response):
        # See: https://roslibpy.readthedocs.io/en/latest/examples/05_subscribe_to_images.html
        # https://answers.ros.org/question/333329/

        img_data = response['color_img']
        img_format = img_data.get('format', 'jpeg')  # or 'png' depending on your system
        img_bytes = img_data['data']

        if len(img_bytes) == 0:
            return response
        
        data = base64.b64decode(img_bytes.encode('ascii'))  # decode base64 to bytes
        np_arr = np.frombuffer(data, dtype=np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        response['color_img'] = img_cv

        return response

    def close(self):
        self.client.terminate()

def main():

    transport = RoslibpyTransport('test_ns')

    request = GymAPIRequest()
    request.header.request_type = RequestType.RESET
    transport.step(request)



if __name__ == '__main__':
    main()
        