#!/usr/bin/env python

import rospy
import tf
import tf.transformations
import numpy as np
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped, Twist, Pose
from franka_msgs.msg import FrankaState
from std_msgs.msg import Float32MultiArray # For potential damping control

class PassiveDSCommander:
    def __init__(self):
        rospy.init_node("passive_ds_commander")

        # Parameters
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "panda_link0")
        self.mocap_world_frame = rospy.get_param("~mocap_world_frame", "mocap_world")
        self.handle_frame_id = rospy.get_param("~handle_frame_id", "mocap_world") # Expected frame of incoming handle pose
        self.handle_topic = rospy.get_param("~handle_topic", "/natnet_ros/DoorHandle/pose")
        self.franka_ee_pose_topic = rospy.get_param("~franka_ee_pose_topic", "/franka_state_controller/ee_pose") # type is geometry_msgs/Pose
        self.twist_command_topic = rospy.get_param("~twist_command_topic", "/passiveDS/desired_twist")
        self.damping_command_topic = rospy.get_param("~damping_command_topic", "/passiveDS/desired_damp_eigval")
        self.debug_target_pose_topic = rospy.get_param("~debug_target_pose_topic", "/passive_ds_commander/target_pose")

        # Control Gains & Limits
        self.linear_gain = rospy.get_param("~linear_gain", 1.5)
        self.angular_gain = rospy.get_param("~angular_gain", 1.0)
        self.max_linear_velocity = rospy.get_param("~max_linear_velocity", 0.5) # m/s
        self.max_angular_velocity = rospy.get_param("~max_angular_velocity", 1.0) # rad/s
        self.publish_rate = rospy.get_param("~publish_rate", 100) # Hz

        # TF Listener
        self.tf_listener = tf.TransformListener()

        # State Variables
        self.current_ee_pose = None # PoseStamped in robot_base_frame
        self.target_ee_pose = None # PoseStamped in robot_base_frame
        self.last_handle_msg_time = None

        # Publishers
        self.twist_pub = rospy.Publisher(self.twist_command_topic, Twist, queue_size=1)
        self.target_pose_pub = rospy.Publisher(self.debug_target_pose_topic, PoseStamped, queue_size=1)
        # Optional: Publisher for damping eigenvalues
        # self.damping_pub = rospy.Publisher(self.damping_command_topic, Float32MultiArray, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.franka_ee_pose_topic, PoseStamped, self.franka_ee_pose_callback)
        rospy.Subscriber(self.handle_topic, PoseStamped, self.door_handle_callback)

        # Wait for TF buffer to fill
        rospy.sleep(1.0)
        rospy.loginfo("PassiveDSCommander initialized.")

    def franka_ee_pose_callback(self, msg):
        # Update the current end-effector pose in the robot base frame
        pose = PoseStamped()
        pose.header.frame_id = self.robot_base_frame # Assuming franka_states publishes O_T_EE relative to base
        pose.header.stamp = rospy.Time.now() # Use current time

        # Position
        pose.pose.position.x = msg.pose.position.x
        pose.pose.position.y = msg.pose.position.y
        pose.pose.position.z = msg.pose.position.z

        # Orientation
        quaternion = msg.pose.orientation
        # Normalize quaternion
        # quaternion = quaternion / np.linalg.norm(quaternion)
        pose.pose.orientation.x = quaternion.x
        pose.pose.orientation.y = quaternion.y
        pose.pose.orientation.z = quaternion.z
        pose.pose.orientation.w = quaternion.w

        self.current_ee_pose = pose

    def door_handle_callback(self, door_handle_pose_mocap):
        # Timestamp for timeout
        self.last_handle_msg_time = rospy.Time.now()

        # Verify frame_id, warn if unexpected
        if not door_handle_pose_mocap.header.frame_id == self.handle_frame_id:
            rospy.logwarn_throttle(5.0, f"Incoming door handle pose has frame_id '{door_handle_pose_mocap.header.frame_id}', expected '{self.handle_frame_id}'. Assuming it's correct and proceeding.")
            # Allow proceeding, but maybe add stricter check depending on requirements
            # return

        try:
            # Ensure transform is available
            self.tf_listener.waitForTransform(self.robot_base_frame, door_handle_pose_mocap.header.frame_id,
                                              door_handle_pose_mocap.header.stamp, rospy.Duration(0.5))
            # Transform the mocap pose to the robot's base frame
            door_handle_pose_base = self.tf_listener.transformPose(self.robot_base_frame, door_handle_pose_mocap)

            # Extract handle pose components in base frame
            handle_pos_base = np.array([door_handle_pose_base.pose.position.x,
                                        door_handle_pose_base.pose.position.y,
                                        door_handle_pose_base.pose.position.z])
            handle_quat_base = np.array([door_handle_pose_base.pose.orientation.x,
                                         door_handle_pose_base.pose.orientation.y,
                                         door_handle_pose_base.pose.orientation.z,
                                         door_handle_pose_base.pose.orientation.w])
            handle_rot_base = R.from_quat(handle_quat_base)

            # --- Calculate Target EE Pose (similar logic to interactive_marker.py) ---
            # Define the desired EE orientation relative to the handle (grasp offset)
            # (X_ee = -Z_handle, Y_ee = -Y_handle, Z_ee = -X_handle)
            rotmat_grasp_offset = np.array( [[0,0,-1],
                                              [0,-1,0],
                                              [-1,0,0]])
            grasp_offset_rot = R.from_matrix(rotmat_grasp_offset)

            # Target EE orientation = Handle orientation * Grasp Offset rotation
            target_rot_base = handle_rot_base * grasp_offset_rot

            # Define the desired EE position offset relative to the handle frame (in handle's coords)
            # Move along the handle's X-axis (which is -Z_ee after offset)
            position_offset_handle_frame = np.array([0.0, 0.0, 0.0]) # this is done in gripper frame pub
            # Transform this offset vector from handle frame to base frame
            position_offset_base_frame = handle_rot_base.apply(position_offset_handle_frame)

            # Target EE position = Handle position + Offset (all in base frame)
            target_pos_base = handle_pos_base + position_offset_base_frame
            # --- End Target EE Pose Calculation ---


            # Update the target_pose state variable
            target_pose_msg = PoseStamped()
            target_pose_msg.header.frame_id = self.robot_base_frame
            target_pose_msg.header.stamp = rospy.Time.now() # Use current time for target
            target_pose_msg.pose.position.x = target_pos_base[0]
            target_pose_msg.pose.position.y = target_pos_base[1]
            target_pose_msg.pose.position.z = target_pos_base[2]
            target_quat_base = target_rot_base.as_quat()
            target_pose_msg.pose.orientation.x = target_quat_base[0]
            target_pose_msg.pose.orientation.y = target_quat_base[1]
            target_pose_msg.pose.orientation.z = target_quat_base[2]
            target_pose_msg.pose.orientation.w = target_quat_base[3]

            self.target_ee_pose = target_pose_msg
            self.target_pose_pub.publish(self.target_ee_pose) # Publish for debugging

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr_throttle(1.0, "TF Error in door_handle_callback: %s", e)
        except Exception as e:
            rospy.logerr_throttle(1.0, "Error in door_handle_callback: %s", e)


    def calculate_and_publish_twist(self):
        if self.current_ee_pose is None or self.target_ee_pose is None:
            # rospy.logwarn_throttle(2.0, "Waiting for current EE pose and target pose.")
            return

        # Check for handle pose timeout
        if self.last_handle_msg_time is None or (rospy.Time.now() - self.last_handle_msg_time) > rospy.Duration(1.0):
             rospy.logwarn_throttle(1.0, "No recent door handle pose received. Stopping robot.")
             # Publish zero twist if handle pose is stale
             zero_twist = Twist()
             self.twist_pub.publish(zero_twist)
             return


        # --- Calculate Pose Error ---
        # Position Error
        current_pos = np.array([self.current_ee_pose.pose.position.x,
                                self.current_ee_pose.pose.position.y,
                                self.current_ee_pose.pose.position.z])
        target_pos = np.array([self.target_ee_pose.pose.position.x,
                               self.target_ee_pose.pose.position.y,
                               self.target_ee_pose.pose.position.z])
        pos_error = target_pos - current_pos

        # Orientation Error (using quaternion difference)
        current_quat = np.array([self.current_ee_pose.pose.orientation.x,
                                 self.current_ee_pose.pose.orientation.y,
                                 self.current_ee_pose.pose.orientation.z,
                                 self.current_ee_pose.pose.orientation.w])
        target_quat = np.array([self.target_ee_pose.pose.orientation.x,
                                self.target_ee_pose.pose.orientation.y,
                                self.target_ee_pose.pose.orientation.z,
                                self.target_ee_pose.pose.orientation.w])

        # Ensure shortest path rotation
        if np.dot(current_quat, target_quat) < 0:
            current_quat = -current_quat

        # Calculate the difference quaternion: q_diff = q_target * q_current_inverse
        # Inverse of a unit quaternion is its conjugate
        current_quat_conj = tf.transformations.quaternion_conjugate(current_quat)
        error_quat = tf.transformations.quaternion_multiply(target_quat, current_quat_conj)

        # Convert error quaternion to axis-angle representation (angle * axis)
        # Angle is 2 * acos(qw)
        # Axis is (qx, qy, qz) / sin(angle/2)
        angle = 2.0 * np.arccos(np.clip(error_quat[3], -1.0, 1.0))
        if abs(angle) < 1e-6:
            axis = np.array([0.0, 0.0, 0.0])
        else:
            # Normalize axis vector
            sin_half_angle = np.sin(angle / 2.0)
            axis = error_quat[:3] / sin_half_angle

            # Ensure angle is in [-pi, pi]
            if angle > np.pi:
                angle -= 2 * np.pi

        # Orientation error vector (rotation vector)
        rot_error = angle * axis
        # --- End Pose Error Calculation ---


        # --- Generate Twist Command ---
        twist_cmd = Twist()

        # Linear velocity (proportional to position error)
        linear_vel = self.linear_gain * pos_error
        # Saturate linear velocity
        linear_vel_norm = np.linalg.norm(linear_vel)
        if linear_vel_norm > self.max_linear_velocity:
            linear_vel = (linear_vel / linear_vel_norm) * self.max_linear_velocity

        twist_cmd.linear.x = linear_vel[0]
        twist_cmd.linear.y = linear_vel[1]
        twist_cmd.linear.z = linear_vel[2]

        # Angular velocity (proportional to orientation error)
        angular_vel = self.angular_gain * rot_error
        # Saturate angular velocity
        angular_vel_norm = np.linalg.norm(angular_vel)
        if angular_vel_norm > self.max_angular_velocity:
            angular_vel = (angular_vel / angular_vel_norm) * self.max_angular_velocity

        twist_cmd.angular.x = angular_vel[0]
        twist_cmd.angular.y = angular_vel[1]
        twist_cmd.angular.z = angular_vel[2]
        # --- End Twist Command Generation ---

        self.twist_pub.publish(twist_cmd)


    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.calculate_and_publish_twist()
            rate.sleep()

if __name__ == "__main__":
    try:
        commander = PassiveDSCommander()
        commander.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in PassiveDSCommander: {e}") 