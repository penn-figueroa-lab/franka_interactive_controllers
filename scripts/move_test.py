#!/usr/bin/env python3
import threading
import time
import os
import sys

import rospy
import numpy as np
import tf
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Pose, PoseStamped


class MoveTestNode:
    def __init__(self):
        rospy.init_node('move_test')
        self.current_state = None

        rospy.Subscriber(
            '/franka_state_controller/franka_states',
            FrankaState,
            self._state_cb,
            queue_size=1
        )
        # publish target pose instead of twist/gripper
        self.pose_pub = rospy.Publisher(
            '/passiveDS/desired_lin_and_ori',
            Pose,
            queue_size=1
        )

        self.damping_pub = rospy.Publisher(
            '/passiveDS/desired_damp_eigval',
            Float32MultiArray,
            queue_size=1)

        rospy.loginfo("Waiting for FrankaState...")
        while not rospy.is_shutdown() and self.current_state is None:
            rospy.sleep(0.1)

        self._spin()

    def _state_cb(self, msg: FrankaState):
        T = np.array(msg.O_T_EE).reshape(4,4).T
        pos = T[:3,3]
        quat = tf.transformations.quaternion_from_matrix(T)
        eul  = R.from_quat(quat).as_euler('xyz', degrees=False)
        self.current_state = {
            "cartesian_position": np.concatenate([pos, eul]),
            "gripper_position": 0.0
        }

    def _spin(self):



        rate = rospy.Rate(50)
        while not rospy.is_shutdown():

            # integrate to get target pose
            curr_pos = self.current_state["cartesian_position"][:3]
            curr_eul = self.current_state["cartesian_position"][3:]
            curr_R   = R.from_euler('xyz', curr_eul, degrees=False)
            

            desired_R = curr_R
            desired_q       = desired_R.as_quat()


            A = -8.0 * np.eye(3)
            target_position = np.array([0.5, 0.0, 0.3])
            desired_vel = np.dot(A, curr_pos - target_position)

            ps = Pose()
            ps.position.x    = float(desired_vel[0])
            ps.position.y    = float(desired_vel[1])
            ps.position.z    = float(desired_vel[2])
            ps.orientation.x = float(desired_q[0])
            ps.orientation.y = float(desired_q[1])
            ps.orientation.z = float(desired_q[2])
            ps.orientation.w = float(desired_q[3])

            self.pose_pub.publish(ps)


            damping = Float32MultiArray()
            damping.data = [30, 30, 30]
            self.damping_pub.publish(damping)



            rate.sleep()


if __name__ == '__main__':
    try:
        MoveTestNode()
    except rospy.ROSInterruptException:
        pass
