class WaypointParams:
    def __init__(self):
        self.kp_scale = 0.0
        self.kd_scale = 0.0
        self.max_velocity = 0.0
        self.max_effort = 0.0

        self.target_link = ""
        self.planner = ""
        self.vel_scale = 0.0
        self.accel_scale = 0.0
        self.blend_radius = 0.0
        self.goal_tol = 0.0
        self.path_tol = 0.0
        self.vertical_angle_scale = 0.0

        self.gripper_width = 0.0

    def to_dict(self):
        return {
            "kp_scale": self.kp_scale,
            "kd_scale": self.kd_scale,
            "max_velocity": self.max_velocity,
            "max_effort": self.max_effort,

            "target_link": self.target_link,
            "planner": self.planner,
            "vel_scale": self.vel_scale,
            "accel_scale": self.accel_scale,
            "blend_radius": self.blend_radius,
            "goal_tol": self.goal_tol,
            "path_tol": self.path_tol,
            "vertical_angle_scale": self.vertical_angle_scale,

            "gripper_width": self.gripper_width,
        }