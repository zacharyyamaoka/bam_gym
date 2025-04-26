class WaypointParams:
    def __init__(self):
        self.kp = 0.0
        self.kd = 0.0
        self.max_velocity = 0.0
        self.max_effort = 0.0
        self.vel_scale = 0.0
        self.accel_scale = 0.0
        self.blend_radius = 0.0
        self.planner = ""
        self.gripper_width = 0.0

    def to_dict(self):
        return {
            "kp": self.kp,
            "kd": self.kd,
            "max_velocity": self.max_velocity,
            "max_effort": self.max_effort,
            "vel_scale": self.vel_scale,
            "accel_scale": self.accel_scale,
            "blend_radius": self.blend_radius,
            "planner": self.planner,
            "gripper_width": self.gripper_width,
        }