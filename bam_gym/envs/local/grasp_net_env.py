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
                 random_scene_order = True,
                 random_ann_order = True,
                 render_mode = None,
                 seed = None):
        """
        Args:
            graspnet_root: Path to the GraspNet dataset root directory.
        """
        super().__init__()
        self.camera = camera
        self.graspnet = GraspNetEval(graspnet_root, camera, split)
        self.render_mode = render_mode
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

    def load_camera_K(self, scene_id):
        """ Camera K is the same per scene, so no need to load it every time """
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

    def load_data(self, scene_id, ann_id) -> dict:
        """
        Loads color, depth, and mask images for a given scene, camera, and annotation id.
        """
        color = self.graspnet.loadRGB(scene_id, self.camera, ann_id)
        depth = self.graspnet.loadDepth(scene_id, self.camera, ann_id)
        mask = self.graspnet.loadMask(scene_id, self.camera, ann_id)

        camera_info = self.load_camera_K(scene_id)
        T_camera_table = self.load_T_camera_table(scene_id, ann_id)
        TIMER.start("plane_2_depth")
        bg_depth = plane_2_depth_vectorized(T_camera_table, self.img_height, self.img_width, camera_info)
        TIMER.stop("plane_2_depth")

        data_dict = {
            "color": color,
            "depth": depth,
            "mask": mask,
            "table_pose": T_camera_table,
            "bg_depth": bg_depth,
            "camera_info": camera_info,
        }

        return data_dict

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

        self.obs = self.load_data(scene_id, ann_id)
        return self.obs, {}

    def step(self, action):
        #TODO right now we just support actions for a single scene annotation... at some point you could test vec env as well
        # You can try multiple actions though and you will get feedback for each one no?
        TIMER.start("step")
        if isinstance(action, np.ndarray):
            actions = [action]
        else:
            actions = list(action)
    
        grasp_group = GraspGroup()

        """
        Grasp()
        - args can be a numpy array or tuple of the score, width, height, depth, rotation_matrix, translation, object_id
        - the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        - the length of the numpy array is 17.
        """
        TIMER.start("creating grasp group")

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
            grasp_params = [score, width, height, depth, rotation_matrix, translation, object_id]

            grasp_group.add(Grasp(*grasp_params))

        TIMER.stop("creating grasp group")

        curr_scene_id = self.scene_order[self.scene_ptr]
        curr_ann_id = self.ann_order[self.ann_ptr]
        TIMER.start("_eval_scene")

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
        TIMER.stop("_eval_scene")

        scene_accuracy, grasp_list_list, score_list_list, collision_list_list = eval_data

        reward = [1.0] # one reward per action

        self.ann_ptr += 1
        if self.ann_ptr >= len(self.ann_order):
            self.reset_ann_order()
            self.scene_ptr += 1
            if self.scene_ptr >= len(self.scene_order):
                self.reset_scene_order()
                self.terminate_flag = True

        terminated = self.terminate_flag
        truncated = False

        # Load next observation using helper
        next_scene_id = self.scene_order[self.scene_ptr]
        next_ann_id = self.ann_order[self.ann_ptr]

        TIMER.start("load_data")
        self.obs = self.load_data(next_scene_id, next_ann_id)
        TIMER.stop("load_data")

        info = {
            "reward_scene_idx": curr_scene_id,
            "reward_ann_idx": curr_ann_id,
            "terminate_flag": self.terminate_flag,
            "table_pose": self.obs["table_pose"],
        }
        del self.obs["table_pose"]

        TIMER.stop("step")

        return self.obs, reward, terminated, truncated, info
    
    def render(self):
        # TODO: visualize scene, grasps, table, etc. with open3d
        pass

    def close(self):
        pass