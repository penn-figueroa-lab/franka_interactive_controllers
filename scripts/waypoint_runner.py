#!/usr/bin/env python3
"""
waypoint_runner_node.py

Publishes a stream of target poses (pos + orientation) obtained by
interpolating through a list of waypoints.  No inverse‑kinematics
is performed – downstream controllers can subscribe to /target_pose
and decide how to realise the motion.
"""

import time
import rospy
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32            # optional gripper channel
from franka_msgs.msg import FrankaState   # for robot pose
import tf.transformations as tft          # quaternion helpers

def transform_action_set_orientation(action_set, transform_matrix):
    """
    Transform the roll, pitch, yaw values in ACTION_SET using the given transformation matrix.
    
    Args:
        action_set: numpy array with shape (N, 6) where columns are [x, y, z, roll, pitch, yaw]
        transform_matrix: 4x4 transformation matrix
        
    Returns:
        transformed_action_set: numpy array with transformed orientations
    """
    transformed_action_set = action_set.copy()
    
    # Extract the 3x3 rotation part of the transformation matrix
    R_transform = transform_matrix[:3, :3]
    
    for i in range(len(action_set)):
        # Convert roll, pitch, yaw to rotation matrix
        R_original = Rotation.from_euler('xyz', action_set[i, 3:6]).as_matrix()

        R_offset = Rotation.from_euler('xyz', [0, 0, 45.0], degrees=True).as_matrix()
        
        # Apply transformation: R_new = R_transform * R_original
        R_transformed = R_offset @ R_transform @ R_original
        
        # Convert back to roll, pitch, yaw
        euler_transformed = Rotation.from_matrix(R_transformed).as_euler('xyz')
        
        # Update the action set
        transformed_action_set[i, 3:6] = euler_transformed
    
    return transformed_action_set


# Define the transformation matrix
TRANSFORM_MATRIX = np.array([
    [1.000, 0.000, 0.000, 0.000],
    [0.000, -1.000, 0.000, 0.000],
    [0.000, 0.000, -1.000, 0.000],
    [0.000, 0.000, 0.000, 1.000]
])


class WaypointRunner:
    def __init__(self):
        # ───────────────────────── parameters ─────────────────────────
        self.waypoints = np.array([[0.6, 0.3, 0.25, 0.0,  0.0,  0.0],
                                    [0.6, 0.3, 0.12, 0.0,  0.0,  0.0],
                                    [0.6, -0.3, 0.12, 0.0,  0.0,  0.0],
                                    [0.6, -0.3, 0.25, 0.0, 0.0,  0.0]])
        self.waypoints = transform_action_set_orientation(self.waypoints, TRANSFORM_MATRIX)
        self.enable_orn       = rospy.get_param("~enable_orientation", True)
        self.enable_gripper   = rospy.get_param("~enable_gripper",   False)
        self.num_interp_pts   = rospy.get_param("~interp_points",    500)
        self.rate_hz          = rospy.get_param("~rate",             100)  # Hz
        self.default_euler    = [0.0, 0.0, 0.0]

        # robot pose tracking
        self.current_pose = None
        self.pose_received = False

        # ───────────────────────── validations ────────────────────────
        self.waypoints = np.asarray(self.waypoints, dtype=np.float64)
        if self.enable_orn:
            expected_cols = 6 + (1 if self.enable_gripper else 0)
        else:
            expected_cols = 3 + (1 if self.enable_gripper else 0)
        if self.waypoints.shape[1] != expected_cols:
            rospy.logfatal("Waypoint dimension mismatch (got %d columns, "
                           "expected %d)", self.waypoints.shape[1], expected_cols)
            raise ValueError("Waypoints format incorrect")


        # publishers
        self.pose_pub    = rospy.Publisher("/franka_right/cartesian_impedance_controller/desired_pose",
                                           PoseStamped,
                                           queue_size=10)
        if self.enable_gripper:
            self.grip_pub = rospy.Publisher("/franka_right/target_gripper",
                                            Int32,
                                            queue_size=10)

        # subscribers
        self.pose_sub = rospy.Subscriber("/franka_right/franka_state_controller/O_T_EE",
                                         PoseStamped,
                                         self.pose_callback,
                                         queue_size=1)

        self.rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("WaypointRunner ready – publishing at %.1f Hz", self.rate_hz)

    def pose_callback(self, msg):
        """Callback for robot pose updates."""
        if not self.pose_received:
            self.current_pose = np.array([msg.pose.position.x,
                                           msg.pose.position.y,
                                           msg.pose.position.z,
                                           msg.pose.orientation.x,
                                           msg.pose.orientation.y,
                                           msg.pose.orientation.z,
                                           msg.pose.orientation.w])
            self.pose_received = True

    # ───────────────────────── interpolation ──────────────────────────
    def interpolate_segment(self, start, end):
        """Return an array of shape (N, 3|6) interpolating start→end."""
        t = np.linspace(0.0, 1.0, self.num_interp_pts)

        # positions – simple lerp
        start_pos, end_pos = start[:3], end[:3]
        positions = (1.0 - t)[:, None] * start_pos + t[:, None] * end_pos

        if not self.enable_orn:
            return positions

        # orientations – no interpolation; keep start orientation for whole segment
        interp_eul = np.tile(end[3:6], (self.num_interp_pts, 1))
        return np.hstack([positions, interp_eul])

    # ───────────────────────── publishing loop ────────────────────────
    def run(self):
        # Wait for initial pose
        rospy.loginfo("Waiting for robot pose...")
        while not self.pose_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        if rospy.is_shutdown():
            return
            
        rospy.loginfo("Robot pose received, starting waypoint execution")
        
        # Extract current pose and add as first waypoint
        current_position = self.current_pose[:3]
        current_rotation = self.current_pose[3:7]  # quaternion (x, y, z, w)
        current_euler = Rotation.from_quat(current_rotation).as_euler('xyz')
        
        current_waypoint = np.concatenate([current_position, current_euler])
        self.waypoints = np.vstack([current_waypoint, self.waypoints])
        
        rospy.loginfo("Added current pose as first waypoint: %s", current_waypoint)
        
        total_segments = len(self.waypoints) - 1
        msg = PoseStamped()
        msg.header.frame_id = "panda_link0"   # use your robot's base frame

        for seg_idx in range(total_segments):
            segment = self.interpolate_segment(self.waypoints[seg_idx],
                                               self.waypoints[seg_idx + 1])

            # gripper value associated with start of segment
            if self.enable_gripper:
                grip_val = int(self.waypoints[seg_idx, -1])

            for pose in segment:
                # fill PoseStamped
                pos = pose[:3]
                if self.enable_orn:
                    eul = pose[3:6]
                else:
                    eul = self.default_euler

                quat = tft.quaternion_from_euler(*eul)

                print(pos)

                msg.header.stamp = rospy.Time.now()
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
                msg.pose.orientation.x, msg.pose.orientation.y,\
                msg.pose.orientation.z, msg.pose.orientation.w = quat

                # publish
                self.pose_pub.publish(msg)
                if self.enable_gripper:
                    self.grip_pub.publish(grip_val)

                self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("waypoint_runner")
    try:
        WaypointRunner().run()
    except rospy.ROSInterruptException:
        pass