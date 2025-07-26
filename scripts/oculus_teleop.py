#!/usr/bin/env python3
import threading
import time
import os
import sys

import rospy
import numpy as np
import tf
from scipy.spatial.transform import Rotation as R
from ppadb.client import Client as AdbClient

from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped

# from oculus_reader.reader import OculusReader

def parse_buttons(text):
    parts = text.split(',')
    buttons = {}
    if 'R' in parts:
        parts.remove('R')
        buttons.update(dict.fromkeys(['A','B','RThU','RJ','RG','RTr'], False))
    if 'L' in parts:
        parts.remove('L')
        buttons.update(dict.fromkeys(['X','Y','LThU','LJ','LG','LTr'], False))
    for k in list(buttons):
        if k in parts:
            buttons[k] = True
            parts.remove(k)
    for item in parts:
        sp = item.split()
        if len(sp) >= 2:
            buttons[sp[0]] = tuple(float(x) for x in sp[1:])
    return buttons

def eprint(*args, **kwargs):
    RED = "\033[1;31m"  
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)

class OculusReader:
    def __init__(self,
            ip_address=None,
            port = 5555,
            APK_name='com.rail.oculus.teleop',
            print_FPS=False,
            run=True
        ):
        self.running = False
        self.last_transforms = {}
        self.last_buttons = {}
        self._lock = threading.Lock()
        self.tag = 'wE9ryARX'

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name
        self.print_FPS = print_FPS
        # if self.print_FPS:
        #     self.fps_counter = FPSCounter()

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self):
        self.stop()

    def run(self):
        self.running = True
        self.device.shell('am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER')
        self.thread = threading.Thread(target=self.device.shell, args=("logcat -T 0", self.read_logcat_by_line))
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system('adb devices')
            client.remote_connect(self.ip_address, self.port)
        device = client.device(self.ip_address + ':' + str(self.port))

        if device is None:
            if retry==1:
                os.system('adb tcpip ' + str(self.port))
            if retry==2:
                eprint('Make sure that device is running and is available at the IP address specified as the OculusReader argument `ip_address`.')
                eprint('Currently provided IP address:', self.ip_address)
                eprint('Run `adb shell ip route` to verify the IP address.')
                exit(1)
            else:
                self.get_device(client=client, retry=retry+1)
        return device

    def get_usb_device(self, client):
        try:
            devices = client.devices()
        except RuntimeError:
            os.system('adb devices')
            devices = client.devices()
        for device in devices:
            if device.serial.count('.') < 3:
                return device
        eprint('Device not found. Make sure that device is running and is connected over USB')
        eprint('Run `adb devices` to verify that the device is visible.')
        exit(1)

    def get_device(self):
        # Default is "127.0.0.1" and 5037
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)

    def install(self, APK_path=None, verbose=True, reinstall=False):
        try:
            installed = self.device.is_installed(self.APK_name)
            if not installed or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'APK', 'teleop-debug.apk')
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                installed = self.device.is_installed(self.APK_name)
                if installed and success:
                    print('APK installed successfully.')
                else:
                    eprint('APK install failed.')
            elif verbose:
                print('APK is already installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    def uninstall(self, verbose=True):
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print('APK uninstall finished.')
                    print('Please verify if the app disappeared from the list as described in "UNINSTALL.md".')
                    print('For the resolution of this issue, please follow https://github.com/Swind/pure-python-adb/issues/71.')
                else:
                    eprint('APK uninstall failed')
            elif verbose:
                print('APK is not installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    @staticmethod
    def process_data(string):
        try:
            transforms_string, buttons_string = string.split('&')
        except ValueError:
            return None, None
        split_transform_strings = transforms_string.split('|')
        transforms = {}
        for pair_string in split_transform_strings:
            transform = np.empty((4,4))
            pair = pair_string.split(':')
            if len(pair) != 2:
                continue
            left_right_char = pair[0] # is r or l
            transform_string = pair[1]
            values = transform_string.split(' ')
            c = 0
            r = 0
            count = 0
            for value in values:
                if not value:
                    continue
                transform[r][c] = float(value)
                c += 1
                if c >= 4:
                    c = 0
                    r += 1
                count += 1
            if count == 16:
                transforms[left_right_char] = transform
        buttons = parse_buttons(buttons_string)
        return transforms, buttons

    def extract_data(self, line):
        output = ''
        if self.tag in line:
            try:
                output += line.split(self.tag + ': ')[1]
            except ValueError:
                pass
        return output

    def get_transformations_and_buttons(self):
        with self._lock:
            return self.last_transforms, self.last_buttons

    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                data = self.extract_data(line)
                if data:
                    transforms, buttons = OculusReader.process_data(data)
                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()

# ——————————————————————————————————————————————————————————————
def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(len(vec)):
        ind = abs(vec[i]) - 1
        X[i, int(ind)] = np.sign(vec[i])
    return X

def euler_to_quat(euler):
    return R.from_euler('xyz', euler, degrees=False).as_quat()

def quat_to_euler(quat):
    return R.from_quat(quat).as_euler('xyz', degrees=False)

def quat_diff(q_current, q_ref):
    r_cur = R.from_quat(q_current)
    r_ref = R.from_quat(q_ref)
    return (r_cur * r_ref.inv()).as_quat()

def add_angles(a, b):
    return a + b

def rmat_to_quat(rmat):
    return R.from_matrix(rmat).as_quat()


class VRPolicy:
    def __init__(
        self,
        right_controller: bool = True,
        max_lin_vel: float = 5.0,
        max_rot_vel: float = 24.0,
        max_gripper_vel: float = 1.0,
        spatial_coeff: float = 1.0,
        pos_action_gain: float = 5.0*20.0,
        rot_action_gain: float = 2.0*40.0,
        gripper_action_gain: float = 3.0,
        rmat_reorder: list = [-2, -1, -3, 4],
    ):
        # instantiate OculusReader
        self.oculus_reader = OculusReader()
        # disable haptics
        self.oculus_reader.device.shell(
            'pm revoke com.rail.oculus.teleop android.permission.VIBRATE'
        )
        self.oculus_reader.device.shell('settings put system haptic_feedback_enabled 0')
        self.oculus_reader.device.shell('svc vibrator cancel')

        self.vr_to_global_mat    = np.eye(4)
        self.max_lin_vel         = max_lin_vel
        self.max_rot_vel         = max_rot_vel
        self.max_gripper_vel     = max_gripper_vel
        self.spatial_coeff       = spatial_coeff
        self.pos_action_gain     = pos_action_gain
        self.rot_action_gain     = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat   = vec_to_reorder_mat(rmat_reorder)
        self.controller_id       = "r" if right_controller else "l"

        self.reset_state()
        t = threading.Thread(target=self._update_internal_state, daemon=True)
        t.start()

    def reset_state(self):
        self._state = {
            "poses": {},
            "buttons": {"A": False, "B": False, "X": False, "Y": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor     = True
        self.reset_orientation = True
        self.reset_origin      = True
        self.robot_origin      = None
        self.vr_origin         = None
        self.vr_state          = None

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read = time.time()
        while True:
            time.sleep(1/hz)
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            # print(buttons)

            if poses == {}:
                continue
            now = time.time()
            self._state["controller_on"] = (now - last_read) < num_wait_sec
            last_read = now

            toggled = (self._state["movement_enabled"] !=
                       buttons[self.controller_id.upper()+"G"])
            self.update_sensor     = self.update_sensor or buttons[self.controller_id.upper()+"G"]
            self.reset_orientation = self.reset_orientation or buttons[self.controller_id.upper()+"J"]
            self.reset_origin      = self.reset_origin or toggled

            self._state["poses"]            = poses
            self._state["buttons"]          = buttons
            self._state["movement_enabled"] = buttons[self.controller_id.upper()+"G"]

            stop_updating = (buttons[self.controller_id.upper()+"J"] or
                             buttons[self.controller_id.upper()+"G"])
            
            if self.reset_orientation:
                rot_mat = np.asarray(poses[self.controller_id])
                if stop_updating:
                    self.reset_orientation = False
                try:
                    rot_mat = np.linalg.inv(rot_mat)
                except Exception:
                    rot_mat = np.eye(4)
                    self.reset_orientation = True
                R_YAW_180 = np.array([[-1.,  0.,  0., 0.],   # 180° about +Z
                                    [ 0., -1.,  0., 0.],
                                    [ 0.,  0.,  1., 0.],
                                    [ 0.,  0.,  0., 1.]])
                # --- flip VR‑room so that +x/+y become –x/–y in robot frame ----
                rot_mat = R_YAW_180 @ rot_mat          #  ←  add this line
                # ---------------------------------------------------------------
                self.vr_to_global_mat = rot_mat

    def _process_reading(self):
        rot_mat = np.asarray(self._state["poses"][self.controller_id])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat

        vr_pos  = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        grip_key = "rightTrig" if self.controller_id == "r" else "leftTrig"
        vr_grip = self._state["buttons"][grip_key][0]

        self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_grip}

    def _limit_velocity(self, lin, rot, grip):
        if np.linalg.norm(lin) > self.max_lin_vel:
            lin *= self.max_lin_vel / np.linalg.norm(lin)
        if np.linalg.norm(rot) > self.max_rot_vel:
            rot *= self.max_rot_vel / np.linalg.norm(rot)
        if abs(grip) > self.max_gripper_vel:
            grip = np.sign(grip) * self.max_gripper_vel
        return lin, rot, grip

    def _calculate_action(self, state_dict):
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False

        rp   = np.array(state_dict["cartesian_position"][:3])
        re   = state_dict["cartesian_position"][3:]
        rq   = euler_to_quat(re)
        rg   = state_dict["gripper_position"]

        if self.reset_origin:
            self.robot_origin = {"pos": rp.copy(), "quat": rq.copy()}
            self.vr_origin    = {"pos": self.vr_state["pos"].copy(),
                                 "quat": self.vr_state["quat"].copy()}
            self.reset_origin = False

        pos_err     = (self.vr_state["pos"] - self.vr_origin["pos"]) - (rp - self.robot_origin["pos"])
        rq_off      = quat_diff(rq, self.robot_origin["quat"])
        vq_off      = quat_diff(self.vr_state["quat"], self.vr_origin["quat"])
        quat_err    = quat_diff(vq_off, rq_off)
        euler_err   = quat_to_euler(quat_err)
        grip_err    = (self.vr_state["gripper"] * 1.5) - rg

        lin_cmd     = pos_err * self.pos_action_gain
        rot_cmd     = euler_err * self.rot_action_gain
        grip_cmd    = grip_err * self.gripper_action_gain

        lin_cmd, rot_cmd, grip_cmd = self._limit_velocity(lin_cmd, rot_cmd, grip_cmd)

        return np.concatenate([lin_cmd, rot_cmd, [grip_cmd]])

    def forward(self, obs_dict):
        if self._state["poses"] == {}:
            print("empty")
            return np.zeros(7)
        return self._calculate_action(obs_dict["robot_state"])


class VRTeleopNode:
    def __init__(self):
        rospy.init_node('oculus_vr_teleop')
        self.policy = VRPolicy()
        self.current_state = None

        rospy.Subscriber(
            '/franka_state_controller/franka_states',
            FrankaState,
            self._state_cb,
            queue_size=1
        )
        # publish target pose instead of twist/gripper
        self.pose_pub = rospy.Publisher(
            '/cartesian_impedance_controller/desired_pose',
            PoseStamped,
            queue_size=1
        )

        rospy.loginfo("Waiting for FrankaState...")
        while not rospy.is_shutdown() and self.current_state is None:
            rospy.sleep(0.1)

        self._spin()

    def _state_cb(self, msg: FrankaState):
        T = np.array(msg.O_T_EE).reshape(4,4).T
        pos = T[:3,3]
        quat = tf.transformations.quaternion_from_matrix(T)
        eul  = quat_to_euler(quat)
        self.current_state = {
            "cartesian_position": np.concatenate([pos, eul]),
            "gripper_position": 0.0
        }

    def _spin(self):

        facing_robot = True


        rate = rospy.Rate(50)
        dt = 1.0 / 50.0
        while not rospy.is_shutdown():
            action = self.policy.forward({"robot_state": self.current_state})
            # print(action)
            # action = [vx, vy, vz, wx, wy, wz, grip]

            # integrate to get target pose
            curr_pos = self.current_state["cartesian_position"][:3]
            curr_eul = self.current_state["cartesian_position"][3:]
            curr_R   = R.from_euler('xyz', curr_eul, degrees=False)

            # if facing_robot:
            #     action[0] = -action[0]
            #     action[1] = -action[1]
            #     action[3] = -action[3]
            #     action[4] = -action[4]
            
            dpos     = action[:3] * dt
            drot     = R.from_rotvec(action[3:6] * dt)
            desired_R = drot * curr_R
            dq       = desired_R.as_quat()
            desired_pos = curr_pos + dpos

            ps = PoseStamped()
            ps.header.stamp = rospy.Time.now()
            ps.header.frame_id = 'world'
            ps.pose.position.x    = float(desired_pos[0])
            ps.pose.position.y    = float(desired_pos[1])
            ps.pose.position.z    = float(desired_pos[2])
            ps.pose.orientation.x = float(dq[0])
            ps.pose.orientation.y = float(dq[1])
            ps.pose.orientation.z = float(dq[2])
            ps.pose.orientation.w = float(dq[3])

            # print(desired_pos)

            self.pose_pub.publish(ps)
            rate.sleep()


if __name__ == '__main__':
    try:
        VRTeleopNode()
    except rospy.ROSInterruptException:
        pass
