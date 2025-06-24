#!/usr/bin/env python3

"""
Don't worry about GymAPI when making this. just make it specificly for Graspnet


Masking the background. I think its a good idea, as I can make sure algo for table fitting is working well...

In here you should not determine how to use the background, just send an accurate depth image of the background...

Just sending the table pose could be enough no? then I can change it manually into a pc? 

I also need to send camera_info! 

From grasp_v5 - grasp_v6 I stop sending a depth image and instead sent the tf for the table! cool function- plane_2_depth()

I had thought about using the grasp representation in bam_ws or python_ws. But what if you need to add more features to it?...

You can use inheritance if you want... but sometimes its nice to just keep things that work isolated.
"""
# BAM

from bam_utils.pointcloud import plane_2_depth_vectorized
from bam_utils.time_helper import Timer

from ros_py_types.bam_srv import GymAPI_Response
from ros_py_types.bam_msgs import GymFeedback, Segment2DArray

from ros_py_types.sensor_msgs import Image, CompressedImage, CameraInfo

from ros_py_types.converter import mask_img_to_segment2D_list

TIMER = Timer(verbose=False)

# PYTHON
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import open3d as o3d
from graspnetAPI import GraspNet, GraspNetEval, GraspGroup, Grasp
import cv2
from transforms3d.euler import euler2mat

class GraspNetEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self,
                 graspnet_root: str,
                 camera = 'realsense',
                 split = 'train',
                 finger_depth = 20/1000, # (meters) from front to back side 
                 finger_height = 100/1000, # (meters) from finger tip to palm
                 finger_thickness = 15/1000, # (meters) from inside surface to outside surface
                 max_width = 0.1, # (meters) max grasp width
                 random_scene_order = False,
                 random_ann_order = False,
                 render_mode = None,
                 vec = False,
                 seed = None):
        """
        Args:
            graspnet_root: Path to the GraspNet dataset root directory.
        """
        super().__init__()
        print("[UNCONFIGURED] GraspNetEnv")

        self.camera = camera
        self.graspnet = GraspNetEval(graspnet_root, camera, split)
        self.graspnet.checkDataCompleteness()
        print("[SUCCESS] Data check complete")
        self.render_mode = render_mode
        self.vec = vec
        self.vis = False
        if self.render_mode == "human":
            self.vis = True

        self.finger_depth = finger_depth
        self.finger_height = finger_height
        self.finger_thickness = finger_thickness
        self.max_width = max_width


        self.random_scene_order = random_scene_order
        self.random_ann_order = random_ann_order
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Prepare scene and annotation indices
        self.scene_ids = list(self.graspnet.sceneIds)
        self.ann_range = list(range(256))  # 256 annotations per scene

        self.scene_order = self.scene_ids.copy()
        self.ann_order = self.ann_range.copy()

        self.scene_ptr = 0
        self.ann_ptr = 0

        self.terminate_flag = False
        self.scene_done = False
        self.ann_done = False

        self.camera_K = None
        self.camera_K_scene = None

        self.obs = None

        self.img_height = 720  # Example image height
        self.img_width = 1280   # Example image width
        # Example observation space: color, depth, mask, table_pose
        self.observation_space = spaces.Dict({
            "color": spaces.Box(0, 255, shape=(720, 1280, 3), dtype=np.uint8),
            "depth": spaces.Box(0, 10000, shape=(720, 1280), dtype=np.uint16),
            "mask": spaces.Box(0, 1, shape=(720, 1280), dtype=np.uint8),
            "table_pose": spaces.Box(-10, 10, shape=(4, 4), dtype=np.float32),
            "bg_depth": spaces.Box(0, 10000, shape=(720, 1280), dtype=np.float32),
            "camera_info": spaces.Box(0, 10000, shape=(3, 3), dtype=np.float32),
        })
        # Example action space: list of grasp poses (x, y, z, rx, ry, rz, grasp_width)
        self.action_space = spaces.Sequence(
            spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        )
        print("[READY] GraspNetEnv")

    def reset_scene_order(self):
        self.scene_order = self.scene_ids.copy()
        if self.random_scene_order:
            self.rng.shuffle(self.scene_order)
        self.scene_ptr = 0

    def reset_ann_order(self):
        self.ann_order = self.ann_range.copy()
        if self.random_ann_order:
            self.rng.shuffle(self.ann_order)
        self.ann_ptr = 0

    def load_camera_K(self, scene_id) -> np.ndarray:
        """ 
        Camera K is the same per scene, so no need to load it every time 
        Format: 
        [[927.16973877   0.         651.31506348]
        [  0.         927.36688232 349.62133789]
        [  0.           0.           1.        ]]
        """
        if scene_id == self.camera_K_scene:
            return self.camera_K        
        else:
            self.camera_K_scene = scene_id
            self.camera_K = self.graspnet.loadCameraK(scene_id, self.camera)
            return self.camera_K

    def load_T_camera_table(self, scene_id, ann_id):
        
        T_table_camera = self.graspnet.loadCameraPose(scene_id, self.camera, ann_id)      
        T_camera_table = np.linalg.inv(T_table_camera)  # Invert to get camera to table pose

        # See how table points are made in graspnetAPI/graspnet_eval.py

        return T_camera_table

    def load_feedback(self, scene_id, ann_id) -> GymFeedback:
        """
        Loads color, depth, and mask images for a given scene, camera, and annotation id.
        """
        color = self.graspnet.loadRGB(scene_id, self.camera, ann_id)
        depth = self.graspnet.loadDepth(scene_id, self.camera, ann_id)/1000 
        mask = self.graspnet.loadMask(scene_id, self.camera, ann_id)

        camera_info = self.load_camera_K(scene_id)

        # can I use these dummy messages?
        # what about when I wrap it in a bam_ros_env
        # Would you ever need to go from dummy_feednacl to real?
        # Lets say I want to test this env but now over a network...
        # at some point I will get a real request. I will unpack it from real_request to dummy
        # and then dummy request to real.. I think that can happen at a later stage though!
        # in a local_dummy_transport, local_ros_transport
        # for testing locally... this is perfect.
        
        color_msg = CompressedImage()
        color_msg.format = "jpeg"
        color_msg.data = color

        depth_msg = Image()
        depth_msg.height = depth.shape[0]
        depth_msg.width = depth.shape[1]
        depth_msg.data = depth

        camera_info_msg = CameraInfo()
        camera_info_msg.k = camera_info.flatten().tolist()  # Flatten to 1D list
        # print("camera_info_msg.k", camera_info_msg.k)

        segments = Segment2DArray()
        # seg_array.header = label_img.header
        segments.segments = mask_img_to_segment2D_list(mask)
        
        # Plot histogram of the mask
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(mask.flatten(), bins=50)
        # plt.title(f"Mask Histogram (scene {scene_id}, ann {ann_id})")
        # plt.xlabel("Mask Value")
        # plt.ylabel("Frequency")
        # plt.show()
        # Values in mask histogram
        # 1 bucket value 0 for background
        # then around 10 other buckets between (0-88) for different objects


        f = GymFeedback()
        f.color_img = [color_msg]
        f.depth_img = [depth_msg]
        f.camera_info = [camera_info_msg]
        f.segments = [segments]

        # T_camera_table = self.load_T_camera_table(scene_id, ann_id)
        # TIMER.start("plane_2_depth")
        # bg_depth = plane_2_depth_vectorized(T_camera_table, self.img_height, self.img_width, camera_info)
        # TIMER.stop("plane_2_depth")


        # Optionally, you can store the original arrays in info for later reconstruction
        # f.info = json.dumps({
        #     "scene_id": scene_id,
        #     "ann_id": ann_id,
        #     "table_pose": T_camera_table.tolist(),
        # })

        # If you want to fill color_img, depth_img, camera_info, etc. as lists:
        # f.color_img = [color]  # If using CompressedImage/Image types, wrap accordingly
        # f.depth_img = [depth]
        # f.camera_info = [camera_info]



        return f

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terminate_flag = False
        self.scene_done = False
        self.ann_done = False

        # Shuffle scene and annotation order if needed
        self.reset_scene_order()
        self.reset_ann_order()

        # Load first observation using helper
        scene_id = self.scene_order[self.scene_ptr]
        ann_id = self.ann_order[self.ann_ptr]

        response = GymAPI_Response()
        response.feedback = [self.load_feedback(scene_id, ann_id)]
        (self.obs, info) = response.to_reset_tuple(vec=self.vec)

        return self.obs, info

    def step(self, action):
        #TODO right now we just support actions for a single scene annotation... at some point you could test vec env as well
        # You can try multiple actions though and you will get feedback for each one no?
        TIMER.start("step")

        #region - Unpack action and create grasp group
        TIMER.start("creating grasp group")

        if isinstance(action, np.ndarray):
            actions = [action]
        else:
            actions = list(action)
    
        grasp_group = GraspGroup()

        for a in actions:
            x, y, z, rx, ry, rz, grasp_width = a

            #TODO add finger thickness option
            score = 0.0  # Placeholder score
            width = grasp_width  # Grasp width from action
            height = self.finger_height
            depth = self.finger_depth
            rotation_matrix = euler2mat(rx, ry, rz, axes='sxyz')  # rotation matrix
            translation = np.array([x, y, z])
            object_id = -1
            # [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
            grasp_params = [score, width, height, depth, rotation_matrix, translation, object_id]
            grasp_group.add(Grasp(*grasp_params))

        TIMER.stop("creating grasp group")
        #endregion - Unpack action and create grasp group

        #region - Evaluate action and determine reward
        TIMER.start("_eval_scene")
        curr_scene_id = self.scene_order[self.scene_ptr]
        curr_ann_id = self.ann_order[self.ann_ptr]

        # TODO edit this to allow for more deterministic evaluation, I should get a result for each grasp I pass in!
        eval_data = self.graspnet._eval_scene(
            scene_id = curr_scene_id,
            ann_id_list = [curr_ann_id], 
            grasp_group_list = [grasp_group],
            TOP_K = len(grasp_group),
            return_list = True,
            vis = self.vis,
            max_width = self.max_width,
            use_cache = True,
            )
        scene_accuracy_list, grasp_list_list, score_list_list, collision_list_list = eval_data

        reward = [1.0] # one reward per action
        TIMER.stop("_eval_scene")
        #endregion - Evaluate action and determine reward

        #region - Tick forward to next scene/annotation and deal with end of episode logic
        self.ann_ptr += 1
        if self.ann_ptr >= len(self.ann_order):
            self.reset_ann_order()
            self.scene_ptr += 1
            if self.scene_ptr >= len(self.scene_order):
                self.reset_scene_order()
                self.terminate_flag = True

        terminated = self.terminate_flag
        truncated = False
        #endregion - Tick forward to next scene/annotation and deal with end of episode logic

        #region - Load next observation using helper
        TIMER.start("load_data")
        next_scene_id = self.scene_order[self.scene_ptr]
        next_ann_id = self.ann_order[self.ann_ptr]
        f = self.load_feedback(next_scene_id, next_ann_id)
        TIMER.stop("load_data")
        #endregion - Load next observation using helper



        # Populate feedback and return tuple
        f.reward = reward
        f.terminated = terminated
        f.truncated = truncated

        response = GymAPI_Response()
        response.feedback = [f]
        (self.obs, reward, terminated, truncated, info) = response.to_step_tuple(vec=self.vec)

        # TODO add this to info dict
        # info = {
        #     "reward_scene_idx": curr_scene_id,
        #     "reward_ann_idx": curr_ann_id,
        #     "terminate_flag": self.terminate_flag,
        #     "table_pose": self.obs["table_pose"],
        # }
        # del self.obs["table_pose"]
        TIMER.stop("step")

        return self.obs, reward, terminated, truncated, info
    
    def render(self):
        # TODO: visualize scene, grasps, table, etc. with open3d
        pass

    def close(self):
        pass

