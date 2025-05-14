import os
import json
import cv2
import datetime
import glob
import numpy as np

class SampleSaver:
    def __init__(self, base_dir, env_name, env_ns="ns", samples_per_json=100000):
        self.base_dir = base_dir
        self.env_name = env_name
        self.env_ns = env_ns
        self.samples_per_json = samples_per_json

        # Make dirs and save the final paths
        self.env_ns_dir = self.make_dirs(base_dir, env_name, env_ns)

        self.color_dir = os.path.join(self.env_ns_dir, "color")
        self.depth_dir = os.path.join(self.env_ns_dir, "depth")
        self.segmentation_dir = os.path.join(self.env_ns_dir, "segmentation")

        self.counter = 0
        self.current_json_idx = 0
        self.current_json_count = 0
        self.sarsa_file = None

        # Try to resume if previous files exist
        self._resume_from_existing()

    def make_dirs(self, base_dir, env_name, env_ns):
        """Create nested save directories: base_dir/env_name/env_ns/{color, depth, segmentation}"""
        env_name_dir = os.path.join(base_dir, env_name)
        os.makedirs(env_name_dir, exist_ok=True)

        env_ns_dir = os.path.join(env_name_dir, env_ns)
        os.makedirs(env_ns_dir, exist_ok=True)

        subfolders = ['color', 'depth', 'segmentation']
        for folder in subfolders:
            os.makedirs(os.path.join(env_ns_dir, folder), exist_ok=True)

        return env_ns_dir

    def _resume_from_existing(self):
        # Look in env_ns_dir, not base_dir
        json_files = sorted(glob.glob(os.path.join(self.env_ns_dir, "sarsa_*.jsonl")))
        if json_files:
            last_file = json_files[-1]
            print(f"[SampleSaver] Resuming from {last_file}")

            self.current_json_idx = int(os.path.splitext(os.path.basename(last_file))[0].split("_")[1])
            with open(last_file, 'r') as f:
                self.current_json_count = sum(1 for _ in f)
            self.counter = self.current_json_idx * self.samples_per_json + self.current_json_count
            self.sarsa_file = os.path.join(self.env_ns_dir, f"sarsa_{self.current_json_idx:04d}.jsonl")

            print(f"[SampleSaver] Current counter: {self.counter}, current json count: {self.current_json_count}")
        else:
            print("[SampleSaver] Starting fresh.")
            self._open_new_sarsa_file()

    def _open_new_sarsa_file(self):
        self.sarsa_file = os.path.join(self.env_ns_dir, f"sarsa_{self.current_json_idx:04d}.jsonl")
        self.current_json_count = 0
        print(f"[SampleSaver] Opened new sarsa file: {self.sarsa_file}")

    def generate_unique_key(self):
        now = datetime.datetime.now()
        timestamp = now.strftime("%y%m%d_%H%M%S")
        milliseconds = int(now.microsecond / 10000)
        key = f"color_{timestamp}{milliseconds:02d}_{self.env_name}"
        return key

    def save_sample(self, observation, action, reward, terminated, truncated, info):
        # Generate unique key
        key = self.generate_unique_key()

        record = {
            '#': self.counter,
            's': dict(),
            'a': None,
            'r': reward,
        }

        # --- Save Action ---
        if isinstance(action, np.ndarray):
            record['a'] = action.tolist()

        elif isinstance(action, (np.integer, np.floating)):
            record['a'] = action.item()

        elif isinstance(action, (tuple, list, int, float)):
            record['a'] = action

        # --- Save observation ---
        if isinstance(observation, np.ndarray):
            record["s"]['obs'] = observation.tolist()
        elif isinstance(observation, (tuple, list, int, float)):
            record["s"]['obs'] = observation
        elif isinstance(observation, (np.integer, np.floating)):
            record["s"]['obs'] = observation.item()

        # --- Save color image ---
        color_img = info.get("color_img", None)
        if hasattr(color_img, "shape"):
            if color_img.shape[-1] == 1:
                color_img = np.repeat(color_img, 3, axis=-1)
            if color_img.dtype != np.uint8:
                color_img = (color_img * 255).clip(0, 255).astype(np.uint8)

            rgb_filename = f"{key}_color.jpg"
            rgb_path = os.path.join(self.color_dir, rgb_filename)
            cv2.imwrite(rgb_path, color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            record['s']["color"] = os.path.relpath(rgb_path, self.env_ns_dir)

        # --- Save depth image ---
        depth_img = info.get("depth_img", None)
        if hasattr(depth_img, "shape"):
            if depth_img.dtype == np.float32 or depth_img.dtype == np.float64:
                depth_vis = (depth_img * 255).clip(0, 255).astype(np.uint8)
            elif depth_img.dtype == np.uint16:
                depth_vis = (depth_img / depth_img.max() * 255).astype(np.uint8)
            else:
                depth_vis = depth_img.astype(np.uint8)

            depth_filename = f"{key}_depth.png"
            depth_path = os.path.join(self.depth_dir, depth_filename)
            cv2.imwrite(depth_path, depth_vis)

            record['s']['depth'] = os.path.relpath(depth_path, self.env_ns_dir)

        # --- Save segmentation image ---
        seg_img = info.get("segmentation_img", None)
        if hasattr(seg_img, "shape"):
            if seg_img.dtype != np.uint8:
                seg_img = (seg_img * 255).clip(0, 255).astype(np.uint8)

            seg_filename = f"{key}_segmentation.png"
            seg_path = os.path.join(self.segmentation_dir, seg_filename)
            cv2.imwrite(seg_path, seg_img)

            record['s']['segmentation'] = os.path.relpath(seg_path, self.env_ns_dir)

        # --- Save SARSA record ---
        with open(self.sarsa_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        self.counter += 1
        self.current_json_count += 1

        # Rotate JSON file if needed
        if self.current_json_count >= self.samples_per_json:
            self.current_json_idx += 1
            self._open_new_sarsa_file()

    def close(self):
        print(f"[SampleSaver] Finished saving {self.counter} samples.")
