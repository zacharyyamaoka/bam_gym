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
from bam_utils.time_helper import StopWatch
from bam_utils.transforms import xyzrpy_to_matrix, matrix_to_xyzrpy
from bam_utils.o3d_helper import O3DViewer


import graspnetAPI.utils
import graspnetAPI.utils.utils
from ros_py_types.bam_srv import GymAPI_Response
from ros_py_types.bam_msgs import GymFeedback, Segment2DArray
from ros_py_types.sensor_msgs import Image, CompressedImage, CameraInfo
from ros_py_types.converter import mask_img_to_segment2D_list

from bam_gym.envs import custom_spaces

TIMER = StopWatch(verbose=False)

# PYTHON
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import open3d as o3d
# careful for bam Grasp and Graspnet Grasp...
import graspnetAPI 
import cv2
from transforms3d.euler import euler2mat
from typing import List

class GraspNetEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}
    
    # TODO right now matching these to the collision values used in Collision checker
    def __init__(self,
                 graspnet_root: str,
                 camera = 'realsense',
                 split = 'train',
                 finger_width = 20/1000, # (meters) from front to back side 
                 finger_height = 60/1000, # (meters) from finger tip to palm
                 finger_thickness = 10/1000, # (meters) from inside surface to outside surface
                 max_width = 0.1, # (meters) max grasp width
                 random_scene_order = False,
                 random_ann_order = False,
                 render_mode = None,
                 vec = False,
                 seed = None,
                 friction_threshold = -1, # grasps above this fail
                 ):
        """
        Args:
            graspnet_root: Path to the GraspNet dataset root directory.
        """
        super().__init__()
        print("[UNCONFIGURED] GraspNetEnv")

        self.camera = camera
        self.root = graspnet_root
        self.graspnet = graspnetAPI.GraspNetEval(graspnet_root, camera, split)
        self.graspnet.checkDataCompleteness()
        print("[SUCCESS] Data check complete")
        self.render_mode = render_mode
        self.vec = vec
        self.vis = False
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = O3DViewer(cam_pos=[0, 0, 0], lookat=[0, 0, 0.4])
            self.vis = True

        self.finger_width = finger_width
        self.finger_height = finger_height
        self.finger_thickness = finger_thickness
        self.max_width = max_width

        self.friction_threshold = friction_threshold

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
        self.observation_space = spaces.Dict({})

        # self.observation_space["obs_names"] = spaces.Sequence(spaces.Text(max_length=32))
        # self.observation_space["obs"] = spaces.Sequence(spaces.Box(low=-1, high=1, shape=None, dtype=np.float32))

        # self.observation_space["pose_names"] = spaces.Sequence(spaces.Text(max_length=32))
        # self.observation_space["pose"] = spaces.Sequence(custom_spaces.pose_space())

        # unpack_len_1_lists = True
        self.observation_space["color_img"] = custom_spaces.color_img_space()
        self.observation_space["depth_img"] = custom_spaces.depth_img_space()
        self.observation_space["camera_info"] = custom_spaces.camera_info_space()
        self.observation_space["segments"] = spaces.Sequence(custom_spaces.segment2darray_space())

        self.action_space = custom_spaces.grasp_array_space()

        if self.vec:
            self.observation_space = spaces.Sequence(self.observation_space)



        # Example action space: list of grasp poses (x, y, z, rx, ry, rz, grasp_width)

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

    def load_graspnet_grasps(self, scene_id, ann_id, fric_coef_thresh=0.2, n=20, sort_by_score=False) -> graspnetAPI.GraspGroup:
        """
        Loads grasps for a given scene and annotation id.
        Returns a list of Grasp objects.
        """
        # Load grasps from GraspNet
        grasps: graspnetAPI.GraspGroup = self.graspnet.loadGrasp(sceneId = scene_id,
                                         annId = ann_id,
                                         format = '6d',
                                         camera = self.camera,
                                         fric_coef_thresh = fric_coef_thresh)
        
        print("Loaded grasps:", len(grasps))
        # selected_grasps = _6d_grasp[:20]
        # _6d_grasp.sort_by_score() # to get best grasps first
        # _6d_grasp.sort_by_score(reverse=True) # to get worst grasps first
        # selected_grasps = _6d_grasp.random_sample(numGrasp = 10)

        # Verify invertable
        test_grasp: graspnetAPI.Grasp = grasps[0]
        xyz, rpy = matrix_to_xyzrpy(test_grasp.tcp_frame)
        action = [xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2], test_grasp.width]
        compare_grasp = self.make_graspnet_grasp(action)

        assert np.allclose(test_grasp.T_graspnet_tcp @ test_grasp.T_tcp_graspnet, np.eye(4)), "T_graspnet_tcp @ T_tcp_graspnet mismatch"
        assert np.allclose(test_grasp.rotation_matrix, compare_grasp.rotation_matrix), "rotation_matrix mismatch"
        assert np.allclose(test_grasp.translation, compare_grasp.translation), "translation mismatch"
        assert np.allclose(test_grasp.width, compare_grasp.width), "width mismatch"
        # assert np.allclose(test_grasp.grasp_array, compare_grasp.grasp_array) # You cannot compare grasp_array directly as scores and object ids are not the same

        if sort_by_score == True:
            grasps.sort_by_score(reverse=True)
            selected_grasps = grasps[:n]
        else:
            selected_grasps = grasps.random_sample(numGrasp=n)

        print("Returning grasps:", len(selected_grasps))

        return selected_grasps
    
    # Its bad design to have these sample them selves as I cannot get any consitency!
    def display_graspnet_grasps(self, grasps: graspnetAPI.GraspGroup, scene_id, ann_id,  use_defaults = True, show_frame = True):
        geometries = []
        geometries.append(self.graspnet.loadScenePointCloud(sceneId = scene_id, annId = ann_id, camera = self.camera))
        geometries += grasps.to_open3d_geometry_list(use_defaults=use_defaults, show_frame=show_frame)
        o3d.visualization.draw_geometries(geometries)

    def graspnet_grasps_to_actions(self, grasps: graspnetAPI.GraspGroup) -> List[np.ndarray]:
        actions = []
        for g in grasps:
            xyz, rpy = matrix_to_xyzrpy(g.tcp_frame)
            grasp_action = np.array([xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2], g.width])
            actions.append(grasp_action)
        return actions

    def make_graspnet_grasp(self, action) -> graspnetAPI.Grasp:

        x, y, z, rx, ry, rz, grasp_width = action

        # Internally graspnet uses a strange coordinate system, see: bam_gym/docs/grasp_net_def.png

        height = self.finger_width
        depth = self.finger_height - 0.02 # minus 2cm to account for depth_base, it will be added back later internally in GraspNet

        T_world_tcp = xyzrpy_to_matrix([x, y, z], [rx, ry, rz])  # Convert to transformation matrix
        grasp_dummy = graspnetAPI.Grasp()
        grasp_dummy.depth = depth
        # T_world_graspnet = T_world_tcp @ graspnetAPI.Grasp().T_tcp_graspnet  #[BUG] you cannot use a blank grasp like this..
        T_world_graspnet = T_world_tcp @ grasp_dummy.T_tcp_graspnet  
        R_world_graspnet = T_world_graspnet[:3, :3] 
        t_world_graspnet = T_world_graspnet[:3, 3]

        score = 0
        object_id = -1

        # [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        grasp_params = [score, grasp_width, height, depth, R_world_graspnet, t_world_graspnet, object_id]
        return graspnetAPI.Grasp(*grasp_params)

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

        info["scene_id"] = scene_id
        info["ann_id"] = ann_id
        return self.obs, info

    def step(self, action: List[np.ndarray], vis=False) -> tuple:
        #TODO right now we just support actions for a single scene annotation... at some point you could test vec env as well
        # You can try multiple actions though and you will get feedback for each one no?
        TIMER.start("step")

        #region - Unpack action and create grasp group
        TIMER.start("creating grasp group")

        # Support passing in a single action
        if not isinstance(action, (list, graspnetAPI.GraspGroup)):
            action = [action]


        # Support directly passing in GraspGroup for debugging
        if isinstance(action, graspnetAPI.GraspGroup):
            grasp_group = action

        else:
            grasp_group = graspnetAPI.GraspGroup()
            for a in action:
                grasp = self.make_graspnet_grasp(a)
                grasp_group.add(grasp)

        TIMER.stop("creating grasp group")
        #endregion - Unpack action and create grasp group

        #region - Evaluate action and determine reward
        TIMER.start("_eval_scene")
        curr_scene_id = self.scene_order[self.scene_ptr]
        curr_ann_id = self.ann_order[self.ann_ptr]

        # TODO edit this to allow for more deterministic evaluation, I should get a result for each grasp I pass in!
        # TODO I could potetially cache all the models once at the start
        # eval_scene_all_grasps
        # eval_data = self.graspnet._eval_scene(
        #     scene_id = curr_scene_id,
        #     ann_id_list = [curr_ann_id], 
        #     grasp_group_list = [grasp_group],
        #     TOP_K = len(grasp_group),
        #     return_list = True,
        #     vis = self.vis,
        #     max_width = self.max_width,
        #     use_cache = True,
        #     )
        eval_data = self.graspnet.eval_scene_all_grasps(curr_scene_id, [curr_ann_id], [grasp_group], vis=vis, use_cache = True)
        grasp_list_list, score_list_list, collision_list_list = eval_data
        scores = score_list_list[0]
        scores_raw = np.copy(scores) 
        collision = collision_list_list[0]

        print("scores: ", scores_raw)

        # Scores are friction coefficients between [1.2, 0.2], Lower is better, -1 means no force closure/collision

        if self.friction_threshold > 0:
            scores[scores > self.friction_threshold] = 0 # no force closure
            scores[scores <= self.friction_threshold] = 1 

        else:
            # Normalize to between [0, 1] for confidence values
            # Optionally threshold to return binary grasp success
            max_score = 1.2
            min_score = 0.2
            scores = -1 * (scores - max_score) / (max_score - min_score)

        scores[scores_raw == -1] = 0 # failed

        print("reward: ", scores)

        self.grasp_group = grasp_group
        self.collision_mask = collision
        self.reward = scores
        self.scores_raw = scores_raw
        self.render() # render the current scene, grasps and rewards, before incrementing the scene pointer

        reward = scores # one reward per action
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

        info["scene_id"] = next_scene_id
        info["ann_id"] = next_ann_id
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

        if self.render_mode != "human":
            return
        
        curr_scene_id = self.scene_order[self.scene_ptr]
        curr_ann_id = self.ann_order[self.ann_ptr]

        # SLOW VERSION CAN OPTIMISE LATER

        color_list = []
        for i in range(len(self.grasp_group)):
            if self.collision_mask[i]: # In collision
                color_list.append((1, 0, 0)) 
            elif self.scores_raw[i] == -1: # No force closure
                color_list.append((0, 0, 0)) 
            else:
                val = self.reward[i]
                r = 1 - val
                g = 1
                b = 1 - val
                color_list.append((r,g,b)) # GREEN to WHITE

        # Yuck! its not easy to add to thr grasp group as it internally manager the grasps as arrays... not classes
        use_defaults = False
        if len(self.grasp_group) > 5:
            use_defaults = True

        # a bit misleading, use_defaults ensures that thin grasps are used. use_collision turns on hardcoded collision values
        # likely if the gripper height and depth are set more reaslitically then if use_collision is false you may still get a thick gripper
        grasps_geometry = self.grasp_group.to_open3d_geometry_list(color_list, use_defaults=use_defaults, use_collision=False, show_frame=True)
        
        t = o3d.geometry.PointCloud()
        t.points = o3d.utility.Vector3dVector(self.graspnet._scene_cache['table_trans'])
        model_list = graspnetAPI.utils.utils.generate_scene_model(self.root, 'scene_%04d' % curr_scene_id , curr_ann_id, return_poses=False, align=False, camera=self.camera)
        pcd = self.graspnet.loadScenePointCloud(curr_scene_id, self.camera, curr_ann_id)
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    
        scene_list = [
            [pcd, *grasps_geometry, origin_frame],
            [pcd, *grasps_geometry, *model_list, origin_frame],
            [*grasps_geometry, *model_list, t, origin_frame]
        ]

        self.viewer.update_scenes(scene_list, reset_view=True)
        self.viewer.run(duration=0, blocking=True)


    def close(self):
        if self.viewer:
            self.viewer.close()

