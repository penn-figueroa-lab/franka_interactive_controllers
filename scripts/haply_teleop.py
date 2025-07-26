#!/usr/bin/env python3
import rospy, numpy as np, tf
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
import HaplyHardwareAPI

def quat_mul(q0, q1):
    w0,x0,y0,z0 = q0; w1,x1,y1,z1 = q1
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1])

def quat_inv(q): q2 = q.copy(); q2[:3]*=-1; return q2 / np.dot(q,q)


# ---------- 1. Handle (verbatim) --------------------------------------------
class Handle(HaplyHardwareAPI.Handle):
    def __init__(self, com):
        HaplyHardwareAPI.Handle.__init__(self, com)
        self.quaternion = None
        self.key        = None

    def OnReceiveHandleInfo(self, data_remaining, device_id, device_model,
                            hardware_version, firmware_version):
        print("Tool info received, device ID:", device_id,
              "device model:", device_model,
              "hardware version:", hardware_version,
              "firmware version:", firmware_version)

    def OnReceiveHandleStatusMessage(self, device_id, quaternion, error_flag,
                                     hall_effect_sensor_level,
                                     user_data_length, user_data):
        self.quaternion = quaternion
        self.key        = user_data[0]

    def OnReceiveHandleErrorResponse(self):
        print("Tool error received")

    def RequestStatus(self):
        HaplyHardwareAPI.Handle.RequestStatus(self)


class HaplyTeleop:
    def __init__(self):
        # discovery / wake-up (verbatim)
        connected_devices  = HaplyHardwareAPI.detect_inverse3s()
        connected_handles  = HaplyHardwareAPI.detect_handles()
        com_stream         = HaplyHardwareAPI.SerialStream(connected_devices[0])
        self.inverse3           = HaplyHardwareAPI.Inverse3(com_stream)
        for k, v in self.inverse3.device_wakeup_dict().items():
            print(k, v)
        handle_stream      = HaplyHardwareAPI.SerialStream(connected_handles[0])
        self.handle             = Handle(handle_stream)
        self.handle.SendDeviceWakeup()
        self.handle.Receive()

        # ---- ROS init ----
        rospy.init_node("haply_tf_teleop")
        self.pose_pub = rospy.Publisher("/haply_pose", PoseStamped, queue_size=1)
        self.br = tf.TransformBroadcaster()
        self.static_br = tf.TransformBroadcaster()

        # ---- State ----
        self.scale = 0.5
        self.pos_d = np.zeros(3)
        self.quat_d = np.array([0,0,0,1])
        self._pressed_prev = 0
        self.prev_pos = np.zeros(3)
        self.prev_quat = np.array([0,0,0,1])


        self.dpdp = np.zeros(3)
        

        # ---- Timers ----
        rospy.Timer(rospy.Duration(0.01), self._static_tf_cb)  # Static tf every 1s
        rospy.Timer(rospy.Duration(0.002), self._poll_cb)     # 500 Hz
        rospy.Timer(rospy.Duration(0.01),  self._ctrl_cb)     # 50 Hz
        rospy.spin()

    def _static_tf_cb(self, _):
        # Publish "world" → "origin" (identity)
        self.static_br.sendTransform(
            (0, 0, 0),
            (0, 0, 0, 1),
            rospy.Time.now(),
            "origin",
            "world"
        )

    def _poll_cb(self, _):
        pos, _ = self.inverse3.end_effector_force(np.zeros(3))
        
        self.handle.RequestStatus()
        self.handle.Receive()

        p = np.array(pos)
        q = np.array(self.handle.quaternion)
        k = self.handle.key



        if k == 1:
            dp = (self.prev_pos - p) * self.scale
            # dq = quat_mul(q, quat_inv(self._q0))
            # dq[2] *= -1  # mirror fix if needed

            if np.max(np.abs(dp)) > 0.0001:
                # arg_dp = np.argmax(np.abs(dp))
                # self.dpdp = np.zeros(3)
                # self.dpdp[arg_dp] = dp[arg_dp]
                self.dpdp = dp

                # self.quat_d = quat_mul(dq, self._q0)

        else:
            self.dpdp = np.zeros(3)

        self._pressed_prev = k
        self.prev_pos = p.copy()
        self.prev_quat = q.copy()

    def _ctrl_cb(self, _):
        now = rospy.Time.now()

        self.pos_d = self.pos_d + self.dpdp
        print("current pos d", self.pos_d, "current dp", self.dpdp)

        # Publish pose
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = "origin"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = self.pos_d
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = self.quat_d
        self.pose_pub.publish(ps)



        # Publish TF from "origin" → "haply"
        self.br.sendTransform(
            self.pos_d,
            self.quat_d,
            now,
            "haply",
            "origin"
        )

if __name__ == "__main__":
    try:
        HaplyTeleop()
    except rospy.ROSInterruptException:
        pass
