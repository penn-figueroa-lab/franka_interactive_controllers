#!/usr/bin/env python3
import rospy
import numpy as np
import tf
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation as R, Slerp
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32MultiArray
from franka_msgs.msg import FrankaState



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
        R_original = R.from_euler('xyz', action_set[i, 3:6]).as_matrix()
        
        # Apply transformation: R_new = R_transform * R_original
        R_transformed = R_transform @ R_original
        
        # Convert back to roll, pitch, yaw
        euler_transformed = R.from_matrix(R_transformed).as_euler('xyz')
        
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

# [x, y, z, roll, pitch, yaw]  (radians)
ACTION_SET_ORIGINAL = np.array([
    [0.4, 0.0, 0.15, 0.0,  0.10,  0.0],
    [0.4, 0.0, 0.08, 0.0,  0.10,  0.0],
    [0.6, 0.0, 0.08, 0.0,  0.10,  0.0],
    [0.6, 0.0, 0.35, 0.0, -0.15,  0.0],
])

# Apply transformation to get the final ACTION_SET
ACTION_SET = transform_action_set_orientation(ACTION_SET_ORIGINAL, TRANSFORM_MATRIX)

def build_path_spline(action_set):
    """Return (pos_spline, rot_slerp, s_grid) from ACTION_SET."""
    xyz = action_set[:, :3]                                # N×3
    seg_lens = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.hstack([0, np.cumsum(seg_lens)])
    s /= s[-1] if s[-1] > 0 else 1.0
    print("s:", s)

    # cubic or lower‑order spline for positions
    pos_spline, _ = splprep([xyz[:, 0], xyz[:, 1], xyz[:, 2]], u=s, k=min(3, len(action_set) - 1), s=0)

    # quaternion Slerp
    quats = R.from_euler('xyz', action_set[:, 3:]).as_quat()
    rot_slerp = Slerp(s, R.from_quat(quats))

    return pos_spline, rot_slerp, s


def project_onto_spline(pos_spline, target, n_samples=200):
    """Closest point parameter s* ∈[0,1] to `target` on the spline."""
    ss = np.linspace(0.0, 1.0, n_samples)
    pts = np.stack(splev(ss, pos_spline), axis=1)
    i = np.argmin(np.linalg.norm(pts - target, axis=1))
    s_init = ss[i]
    eps = 1e-4
    p = np.array(splev(s_init, pos_spline)).T.squeeze()
    dp = (np.array(splev(min(s_init + eps, 1.0), pos_spline)).T.squeeze()
          - np.array(splev(max(s_init - eps, 0.0), pos_spline)).T.squeeze()) / (2 * eps)
    s_ref = s_init - np.dot(p - target, dp) / (np.linalg.norm(dp) ** 2 + 1e-9)
    return np.clip(s_ref, 0.0, 1.0)


def publish_path_visual(path_pub, pos_spline, rot_slerp, frame_id="panda_link0",
                        n_samples=200):
    """Publish a dense PoseArray representing the spline for RViz."""
    pa = PoseArray()
    pa.header.stamp = rospy.Time.now()
    pa.header.frame_id = frame_id

    ss = np.linspace(0.0, 1.0, n_samples)
    pts = np.stack(splev(ss, pos_spline), axis=1)
    quats = rot_slerp(ss).as_quat()
    for i in range(n_samples):
        p = pts[i]
        quat = quats[i]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, p)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = \
            map(float, quat)
        pa.poses.append(pose)

    path_pub.publish(pa)
    rospy.loginfo("Published path spline to /gfabric_path (%d poses)", n_samples)


# ───────────────────────────── hyper‑parameters ─────────────────
LAMBDA_PAR = 10.0          # cost along tangent
LAMBDA_NOR = 3.0         # cost orthogonal
BETA_FAR   = 0.15         # (m) influence radius
K_R        = 4.0          # orientation gain (rad/s)
MAX_LIN_V  = 0.45         # (m/s) velocity clip
MAX_ANG_V  = 1.0          # (rad/s) angular velocity clip
PUB_RATE   = 100          # (Hz)
LOOKAHEAD = 0.07          # metres ahead of s*   (tune 2-5 cm)
GAIN      = 90.0           # 1/s – converts gap → velocity


# ───────────────────────────── main node class ─────────────────
class GFabricVelNode:
    def __init__(self):
        rospy.init_node('gfabric_move_ee')

        # prepare spline and path visual
        self.pos_spline, self.rot_slerp, self.s_wp = build_path_spline(ACTION_SET)
        self.quat_table = R.from_euler('xyz', ACTION_SET[:, 3:]).as_quat()
        self.path_pub = rospy.Publisher('/gfabric_path',
                                        PoseArray, queue_size=1, latch=True)
        self.interpolate_samples = 200
        self.s_grid = np.linspace(0.0, 1.0, self.interpolate_samples)
        self.quat_grid = np.array([self.rot_slerp(s).as_quat() for s in self.s_grid])
        publish_path_visual(self.path_pub, self.pos_spline, self.rot_slerp)

        # Franka state
        self.current_pos, self.current_rot = None, None
        rospy.Subscriber('/franka_state_controller/franka_states',
                         FrankaState, self._state_cb, queue_size=1)

        # command publishers
        self.pose_pub = rospy.Publisher('/passiveDS/desired_lin_and_ori',
                                        Pose, queue_size=1)
        self.damping_pub = rospy.Publisher('/passiveDS/desired_damp_eigval',
                                           Float32MultiArray, queue_size=1)

        rospy.loginfo("Waiting for FrankaState …")
        while not rospy.is_shutdown() and self.current_pos is None:
            rospy.sleep(0.1)
        rospy.loginfo("FrankaState received. Running …")
        self._spin()

    # ───── callbacks ─────
    def _state_cb(self, msg: FrankaState):
        T = np.array(msg.O_T_EE).reshape(4, 4).T
        self.current_pos = T[:3, 3]
        quat = tf.transformations.quaternion_from_matrix(T)
        self.current_rot = R.from_quat(quat)

    # ───── main loop ─────
    def _spin(self):
        rate = rospy.Rate(PUB_RATE)
        while not rospy.is_shutdown():
            self._step()
            rate.sleep()

    def _step(self):
        if self.current_pos is None:
            return

        x = self.current_pos
        R_curr = self.current_rot

        # 1. projection
        s_star = project_onto_spline(self.pos_spline, x, n_samples=self.interpolate_samples)
        p_s = np.array(splev(s_star, self.pos_spline)).T.squeeze()

        # tangent vector
        ds = 1e-4
        t = (np.array(splev(min(s_star + ds, 1.0), self.pos_spline)).T.squeeze()
             - np.array(splev(max(s_star - ds, 0.0), self.pos_spline)).T.squeeze())
        t /= np.linalg.norm(t) + 1e-9

        # 2. metric M(x)
        n1 = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n1 -= n1.dot(t) * t
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(t, n1)
        B = np.stack([t, n1, n2], axis=1)

        dist = np.linalg.norm(x - p_s)
        w = np.exp(- (dist / BETA_FAR) ** 2)
        Λ = np.diag([LAMBDA_PAR,
                     (1 - w) * 1.0 + w * LAMBDA_NOR,
                     (1 - w) * 1.0 + w * LAMBDA_NOR])
        M = B @ Λ @ B.T

        # 3. translational velocity
        v_lin = -M.dot(x - p_s)

        L_total = 1.0
        s_cmd = min(s_star + LOOKAHEAD / L_total, 1.0)
        p_cmd = np.array(splev(s_cmd, self.pos_spline))
        v_lin += GAIN * (p_cmd - x)

        # Scale max velocity based on distance to goal to prevent oscillation
        goal_pos = np.array(splev(1.0, self.pos_spline))
        dist_to_goal = np.linalg.norm(goal_pos - x)
        print("Distance to goal: %.3f m" % dist_to_goal)
        SLOWDOWN_THRESHOLD = 0.04  # Start slowing down when within 3cm of goal
        MIN_VELOCITY_SCALE = 0.8  # Minimum velocity scale factor
        
        if dist_to_goal < SLOWDOWN_THRESHOLD:
            # Exponential decay: slow at beginning, fast at end
            normalized_dist = dist_to_goal / SLOWDOWN_THRESHOLD  # 0 to 1
            velocity_scale = max(MIN_VELOCITY_SCALE, normalized_dist ** 3)
            scaled_max_lin_v = MAX_LIN_V * velocity_scale
        else:
            scaled_max_lin_v = MAX_LIN_V

        speed = np.linalg.norm(v_lin)
        if speed > scaled_max_lin_v:
            v_lin *= scaled_max_lin_v / speed

        idx = np.argmin(np.abs(self.s_grid - s_cmd))      # integer 0 … ORI_SAMPLES-1
        q_des = self.quat_grid[idx]                       # (x, y, z, w)
        # R_des = R.from_quat(q_des)

        # rot_err_vec = (R_curr.inv() * R_des).as_rotvec()  # SO(3) log
        # v_ang = K_R * rot_err_vec
        # if np.linalg.norm(v_ang) > MAX_ANG_V:
        #     v_ang *= MAX_ANG_V / np.linalg.norm(v_ang)

        # 5. publish velocity & orientation

        print("s*: %.3f, dist: %.3f, v_lin: [%.3f, %.3f, %.3f]" %
              (s_star, dist, v_lin[0], v_lin[1], v_lin[2]))
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = map(float, v_lin)

        #q_des = [1.0, 0.0, 0.0, 0.0]  # default quaternion
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = \
            map(float, q_des)
        self.pose_pub.publish(msg)

        # constant damping eigen‑values
        self.damping_pub.publish(Float32MultiArray(data=[30.0, 30.0, 30.0]))


# ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        GFabricVelNode()
    except rospy.ROSInterruptException:
        pass