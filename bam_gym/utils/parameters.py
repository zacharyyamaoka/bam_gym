#!/usr/bin/env python3

from ros_py_types.bam_msgs import WaypointParams

"""
Copied from bam_moveit/bam_moveit_client/bam_moveit_client/msg_factory.py

PLANNERS = {
    'lin': ('pilz_industrial_motion_planner', 'LIN'),
    'ptp':  ('pilz_industrial_motion_planner', 'PTP'),
    'cartesian': ('ompl', 'CartesianPath'),
    'rrt': ('ompl', 'RRTConnectkConfigDefault'),
    'stomp': ('stomp', 'stomp_planner'),
}

"""


def get_default_params(gripper_width=0.0):

    p = WaypointParams()

    p.kp_scale = 1.0
    p.kd_scale = 1.0
    p.max_velocity = 10.0
    p.max_effort = 1.0

    p.target_link = "tcp_world"
    p.planner = "ptp"
    p.vel_scale = 1.0
    p.accel_scale = 1.0
    p.blend_radius = 0.0
    p.goal_tol = 0.2
    p.path_tol = 0.2
    p.vertical_angle_scale = 0.0

    p.gripper_width = gripper_width

    return p
