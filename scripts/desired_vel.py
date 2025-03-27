#!/usr/bin/python3
import rospy, yaml, numpy as np, os, json
import rospkg
from catkin.find_in_workspaces import find_in_workspaces
from IPython import embed
import itertools
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Twist, Pose, PoseStamped
from tf.transformations import quaternion_from_matrix
from nav_msgs.msg import Path

import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax  # https://github.com/deepmind/optax
# from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate

import equinox.experimental as eqxe
import jax.tree_util as jtu
import functools as ft

import cvxpy as cp

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import bagpy
from bagpy import bagreader  # this contains a class that does all the hard work of reading bag files


# from sklearn.preprocessing import MinMaxScaler
# import plotly.graph_objects as go


# ROS node reference Tianyu: https://github.com/tonylitianyu/golfbot/blob/master/nodes/moving#L63

def read_data_csv(csv_files_all, start_indx_all, end_indx_all):
    traj_all = []
    time_all = []

    for csv in csv_files_all:
        ee_states = pd.read_csv(csv)

        traj_all.append(
            np.array([ee_states['pose.position.x'], ee_states['pose.position.y'], ee_states['pose.position.z']]).T)

        time_all.append(np.array(ee_states['Time'] - ee_states['Time'][0]).reshape((-1, 1)))

        # ax.plot(traj_all[-1][:end_indx, 0], traj_all[-1][:end_indx, 1], traj_all[-1][:end_indx, 2])

    traj_c = len(traj_all)

    traj_all_norm = []
    time_all_norm = []
    scaler_all = []
    scaler_all_t = []
    for i in range(traj_c):
        start_indx = int(start_indx_all[i])
        end_indx = int(traj_all[i].shape[0] - end_indx_all[i])
        traj_all[i] = traj_all[i][start_indx:end_indx]
        time_all[i] = time_all[i][start_indx:end_indx] - time_all[i][start_indx]
        scaler_all_t.append(1 / time_all[i][-1])
        tsi = time_all[i] / time_all[i][-1]
        time_all_norm.append(tsi)

    # for i in range(traj_c):
    #     ax.plot(traj_all_norm[i][0, 0], traj_all_norm[i][0, 1], traj_all_norm[i][0, 2], 'ro')
    #     ax.plot(traj_all_norm[i][:, 0], traj_all_norm[i][:, 1], traj_all_norm[i][:, 2])
    #     ax.plot(traj_all_norm[i][-1, 0], traj_all_norm[i][-1, 1], traj_all_norm[i][-1, 2], 'go')

    dim = traj_all[0].shape[1]

    nsamples = 300
    ts_new = np.linspace(0, 1, nsamples)

    traj_all_process = jnp.zeros((traj_c, nsamples, dim))

    traj_all_t_norm = []

    for i in range(traj_c):
        for j in range(dim):
            f = interpolate.interp1d(time_all_norm[i][:, 0], traj_all[i][:, j])
            # f = interpolate.interp1d(time_all[i][:, 0], traj_all[i][:, j])
            # ts_new = np.linspace(time_all[i][0, 0], time_all[i][-1, 0], nsamples)
            # time_all_process = time_all_process.at[i].set(ts_new)
            traj_new = f(ts_new)
            traj_all_process = traj_all_process.at[i, :, j].set(traj_new)

    traj_d = jnp.diff(traj_all_process, axis=1)
    ts_d = jnp.diff(ts_new, axis=0)
    traj_vel = traj_d / ts_d[:, np.newaxis]
    # traj_vel = traj_d
    # traj_d = jnp.concatenate((traj_d, jnp.zeros((traj_c, 1, dim))), axis=1)
    # traj_d_all = jnp.concatenate((traj_all_process, traj_d), axis=2)
    traj_vel = jnp.concatenate((traj_vel, jnp.zeros((traj_c, 1, dim))), axis=1)
    traj_vel_all = jnp.concatenate((traj_all_process, traj_vel), axis=2)

    return traj_all_process, traj_vel_all, ts_new, scaler_all_t


class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initializer = jnn.initializers.orthogonal()
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        model_key = key
        key_weights = jrandom.split(model_key, depth + 1)

        for i in range(depth + 1):
            where = lambda m: m.layers[i].weight
            shape = self.mlp.layers[i].weight.shape
            self.mlp = eqx.tree_at(where, self.mlp, replace=initializer(key_weights[i], shape, dtype=jnp.float32))

    def __call__(self, t, y, args):
        return self.mlp(y)

        # model_with_sn = apply_sn(self.mlp)

        # return self.mlp(jnp.concatenate([y, jnp.array([t])]))


class Funcd(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=2 * data_size,
            out_size=2 * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, yd, args):
        return self.mlp(yd)
        # return self.mlp(jnp.concatenate([yd, jnp.array([t])]))


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, yd0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=yd0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


class NeuralODEd(eqx.Module):
    func: Funcd

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Funcd(data_size, width_size, depth, key=key)

    def __call__(self, ts, yd0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=yd0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


def load_model(traj, time, model_file, d_flag):
    ys = traj
    ts = time
    _, length_size, data_size = ys.shape
    width_size = 128
    depth = 3
    seed = 1000
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)
    if d_flag == 0:
        model1 = NeuralODE(data_size, width_size, depth, key=model_key)
    else:
        model1 = NeuralODEd(data_size, width_size, depth, key=model_key)
    model_load = eqx.tree_deserialise_leaves(model_file, model1)

    return model_load


# path = Path()
# path_r = Path()
# path_g = Path()


class desired_vel(object):

    def __init__(self, traj_all_process, time, scaler_all_t, model_load):
        self.traj_all_process = traj_all_process
        self.time = time
        self.model_load = model_load
        self.scaler_all_t = scaler_all_t
        self.pub = rospy.Publisher('/cartesian_impedance_controller/desired_twist', Twist, queue_size=10)
        # self.prevs_O_T_EE = np.zeros((4,4))
        # self.state_sub = rospy.Subscriber("/franka_state_controller/O_T_EE", PoseStamped, self.franka_callback,
        #                                   queue_size=1,
        #                                   tcp_nodelay=True)
        # self.state_data = PoseStamped()
        self.state_sub = rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_callback,
                                          queue_size=1,
                                          tcp_nodelay=True)
        self.state_data = FrankaState()
        self.path_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.path_pub_r = rospy.Publisher('/path_r', Path, queue_size=10)
        self.path_pub_g = rospy.Publisher('/path_g', Path, queue_size=10)
        self.f1x = np.zeros((3,))
        self.xref_g_i = 0 # Initialize index of purely time parameterized reference trajectory
        self.indx = 0 # which initial condition from training data to generate reference trajectory
        self.xref = self.model_load(self.time, self.traj_all_process[self.indx, 0, :])
        self.xref_vel = jax.vmap(self.model_load.func.mlp, in_axes=0)(self.xref)
        self.f1x_start = False
        self.ind_range = 7 # look how far around closest x_ref
        self.ball_xref = 0.1 # look how far around currrent x_t
        self.vopt_max = 0.5 # maximum virtual control input 

    @staticmethod
    def q_from_R(R):
        """ generates quaternion from 3x3 rotation matrix """
        _R = np.eye(4)
        _R[:3, :3] = R
        return quaternion_from_matrix(_R)

    def franka_callback(self, state_msg):
        self.state_data = state_msg

        ## Publish path

        # global path
        # path.header = self.state_data.header
        # pose = PoseStamped()
        # pose.header = self.state_data.header
        # pose.pose = self.state_data.O_T_EE.pose
        # path.poses.append(pose)
        # self.path_pub.publish(path)

        ## Publish path from FrankaState

        # global path

        # path.header.stamp = state_msg.header.stamp
        # path.header.frame_id = "panda_link0"
        # O_T_EE = np.array(self.state_data.O_T_EE).reshape(4, 4).T
        # quat_ee = self.q_from_R(O_T_EE[:3, :3])
        # pose = PoseStamped()
        # pose.header.stamp = state_msg.header.stamp
        # pose.header.frame_id = "panda_link0"
        # pose.pose.position.x = O_T_EE[0, 3]
        # pose.pose.position.y = O_T_EE[1, 3]
        # pose.pose.position.z = O_T_EE[2, 3]
        # pose.pose.orientation.x = quat_ee[0]
        # pose.pose.orientation.y = quat_ee[1]
        # pose.pose.orientation.z = quat_ee[2]
        # pose.pose.orientation.w = quat_ee[3]
        # path.poses.append(pose)
        # self.path_pub.publish(path)

    def cmd_desired_vel(self):
        # vel = O_T_EE - self.prevs_O_T_EE
        # self.prevs_O_T_EE = O_T_EE
        # x = self.state_data.pose.position.x
        # y = self.state_data.pose.position.y
        # z = self.state_data.pose.position.z

        O_T_EE = np.array(self.state_data.O_T_EE).reshape(4, 4).T
        x = O_T_EE[0, 3]
        y = O_T_EE[1, 3]
        z = O_T_EE[2, 3]

        quat_ee = self.q_from_R(O_T_EE[:3, :3])

        x_t = np.array([x, y, z])
        ys = self.traj_all_process
        # f = lambda z : self.model_load.func(None, z, None)

        vopt_bound = self.vopt_max * (1 / self.scaler_all_t[self.indx])

        fx = self.model_load.func.mlp(x_t)

        # xref = self.model_load(self.time, ys[indx, 0, :])

        # time_xref_indx = np.argmin(np.abs(np.linspace(0, 1, xref.shape[0]) - (self.state_data.time - self.start_time) * (self.scaler_all_t[indx])))

        # import ipdb; ipdb.set_trace()

        xref_vel_u = self.xref_vel / jnp.linalg.norm(self.xref_vel, axis=1).reshape((-1, 1))

        xref_vel_all = np.hstack((self.xref[:, :2], self.xref_vel[:, :2]))

        fx_t = np.asarray(fx)

        alpha_h = 10
        gamma = 0.1
        lambda_v = 0

        # r = 0.04  # CBF radius TB
        r = 0.05  # CBF radius LR

        if self.xref_g_i >= self.xref.shape[0]:
            self.xref_g_i = self.xref.shape[0] - 1
        else:
            self.xref_g_i += 1

        xref_i = self.xref[self.xref_g_i, :]
        
        if np.linalg.norm(self.xref[-1, :] - x_t) < 2 * r:
            # desired_twist = Twist()
            #
            # desired_twist.linear.x = 0
            # desired_twist.linear.y = 0
            # desired_twist.linear.z = 0
            # desired_twist.angular.x = 0
            # desired_twist.angular.y = 0
            # desired_twist.angular.z = 0
            #
            # self.pub.publish(desired_twist)

            xref_t = self.xref[-1, :]

            fxref = self.model_load.func.mlp(xref_t)
            fxref_t = np.asarray(fxref)

            alpha_h = 100

            h = -2 * ((x_t - xref_t).T @ (fx_t - fxref_t)) + alpha_h * (
                    -((x_t - xref_t).T @ (x_t - xref_t)) + r ** 2)  # CBF
            
            Q = np.eye(x_t.shape[0])
            G = 2 * (x_t - xref_t).T

            vopt = cp.Variable(x_t.shape[0])
            prob = cp.Problem(cp.Minimize(cp.quad_form(vopt, Q) + lambda_v * cp.pos(G @ vopt - h)), [G @ vopt <= h, vopt <= vopt_bound, -vopt <= vopt_bound])

            prob.solve()

            vopt_chosen = vopt.value
        else:

            ## command to go to closest xref => xref_t
            if not self.f1x_start:
                self.f1x = fx_t
                self.f1x_start = True
            x_t_all = np.hstack((x_t[:2], self.f1x[:2]))
            f1x_u = self.f1x / np.linalg.norm(self.f1x)
            cos_angle = np.clip(np.dot(xref_vel_u, f1x_u), -1, 1)
            dist_ref = np.linalg.norm(self.xref - x_t, axis=1)
            dist_ref_u = dist_ref / np.max(dist_ref)
            # xref_t = xref[np.argmin(np.linalg.norm(xref_vel_all - x_t_all, axis=1)), :]

            # xref_t = self.xref[np.argmax(0 * cos_angle - 1 * dist_ref_u), :]
            # xref_t = xref[np.argmin(np.linalg.norm(xref - x_t, axis=1)), :]
            # xref_t = xref[time_xref_indx, :]

            # xref_t = xref_i
            
            closest_ind = np.argmax(0 * cos_angle - 1 * dist_ref_u)

            # chosen_ind = np.where(dist_ref_u <= self.ball_xref)[0]

            ## looking forward

            chosen_ind_all = np.arange(closest_ind, closest_ind+self.ind_range, 1, dtype=int)
            chosen_ind = np.clip(chosen_ind_all, 0, self.xref.shape[0]-1)

            if chosen_ind.shape[0] == 0:
                xref_ind = closest_ind
                xref_t = self.xref[xref_ind, :]

                fxref = self.model_load.func.mlp(xref_t)
                fxref_t = np.asarray(fxref)

                h = -2 * ((x_t - xref_t).T @ (fx_t - fxref_t)) - alpha_h * ((x_t - xref_t).T @ (x_t - xref_t))  # CLF

                Q = np.eye(x_t.shape[0])
                G = 2 * (x_t - xref_t).T

                vopt = cp.Variable(x_t.shape[0])
                prob = cp.Problem(cp.Minimize(cp.quad_form(vopt, Q) + lambda_v * cp.pos(G @ vopt - h)), [G @ vopt <= h, vopt <= vopt_bound, -vopt <= vopt_bound])

                prob.solve()

                vopt_chosen = vopt.value

            else:

            # min_ind = np.max((int(closest_ind - 1), 0))
            # max_ind = np.min((closest_ind + self.ind_range, self.xref.shape[0]))
            # chosen_ind = np.arange(min_ind, max_ind)

                for i in chosen_ind:
                    xref_t = self.xref[i, :]

                    fxref = self.model_load.func.mlp(xref_t)
                    fxref_t = np.asarray(fxref)

                    h = -2 * ((x_t - xref_t).T @ (fx_t - fxref_t)) - alpha_h * ((x_t - xref_t).T @ (x_t - xref_t))  # CLF

                    Q = np.eye(x_t.shape[0])
                    G = 2 * (x_t - xref_t).T

                    vopt = cp.Variable(x_t.shape[0])
                    prob = cp.Problem(cp.Minimize(cp.quad_form(vopt, Q) + lambda_v * cp.pos(G @ vopt - h)), [G @ vopt <= h, vopt <= vopt_bound, -vopt <= vopt_bound])

                    prob.solve()

                    if i == chosen_ind[0]:
                        vopt_chosen = vopt.value
                        vopt_norm = np.linalg.norm(vopt_chosen)
                        xref_ind = i

                    else:
                        if np.linalg.norm(vopt.value) <= vopt_norm:
                            vopt_norm = np.linalg.norm(vopt.value)
                            vopt_chosen = vopt.value
                            xref_ind = i           

            xref_t = self.xref[xref_ind, :]


        ## Publish reference trajectory

        # global path_r

        # path_r.header.stamp = self.state_data.header.stamp
        # path_r.header.frame_id = "panda_link0"
        # pose_r = PoseStamped()
        # pose_r.header = self.state_data.header
        # pose_r.pose.position.x = xref_t[0]
        # pose_r.pose.position.y = xref_t[1]
        # pose_r.pose.position.z = xref_t[2]
        # pose_r.pose.orientation.x = quat_ee[0]
        # pose_r.pose.orientation.y = quat_ee[1]
        # pose_r.pose.orientation.z = quat_ee[2]
        # pose_r.pose.orientation.w = quat_ee[3]
        # path_r.poses.append(pose_r)
        # self.path_pub_r.publish(path_r)

        # global path_g

        # path_g.header.stamp = self.state_data.header.stamp
        # path_g.header.frame_id = "panda_link0"
        # pose_g = PoseStamped()
        # pose_g.header = self.state_data.header
        # pose_g.pose.position.x = xref_i[0]
        # pose_g.pose.position.y = xref_i[1]
        # pose_g.pose.position.z = xref_i[2]
        # pose_g.pose.orientation.x = quat_ee[0]
        # pose_g.pose.orientation.y = quat_ee[1]
        # pose_g.pose.orientation.z = quat_ee[2]
        # pose_g.pose.orientation.w = quat_ee[3]
        # path_g.poses.append(pose_g)
        # self.path_pub_g.publish(path_g)

        # h = -2*(((x_t-xref_t).T)@(fx_t - fxref_t)) + alpha_h*(-(((x_t-xref_t).T)@(x_t-xref_t)) + r**2) # CBF
        # h = -(alpha_h*(-(((x1-xref_t).T)@(x1-xref_t)) + r**2) + gamma)

        # h = alpha_h*(((x1-xref_t).T)@(x1-xref_t))+gamma

        # h = alpha_h*(-(((x1-xref_t).T)@(x1-xref_t)) + r**2) + gamma

        # import ipdb; ipdb.set_trace()

        # f1 = lambda z : model_load.func(None, z, None) + vopt.value

        self.f1x = self.model_load.func.mlp(x_t) + vopt_chosen

        # f1x = f1(x)

        desired_twist = Twist()

        # elif self.stop:
        #
        #     desired_twist.linear.x = 0
        #     desired_twist.linear.y = 0
        #     desired_twist.linear.z = 0
        #     desired_twist.angular.x = 0
        #     desired_twist.angular.y = 0
        #     desired_twist.angular.z = 0

        k = 2.8

        desired_twist.linear.x = self.f1x[0] * self.scaler_all_t[self.indx] * k
        desired_twist.linear.y = self.f1x[1] * self.scaler_all_t[self.indx] * k
        desired_twist.linear.z = self.f1x[2] * self.scaler_all_t[self.indx] * k
        desired_twist.angular.x = 0
        desired_twist.angular.y = 0
        desired_twist.angular.z = 0

        self.pub.publish(desired_twist)

def main():
    ## Load data
    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    # New demos

    ## LR traj
    # csv_files_all = [
    #     curr_path + '/config/Data_Trajs/demo_2023-04-18-13-26-55/franka_state_controller-O_T_EE.csv']
    # start_indx_all = np.array([300])
    # end_indx_all = np.array([600])  # substract from end time

    ## TB traj
    # csv_files_all = [
    #     '/home/farhadnawaz/catkin_ws/src/franka_interactive_controllers/config/New_demonstrations/demo_2023-04-18-13-28-34/franka_state_controller-O_T_EE.csv']
    # start_indx_all = np.array([400])
    # end_indx_all = np.array([400])  # substract from end time

    ## Old demos

    # LR
    csv_files_all = [
        curr_path + '/config/Data_Trajs/demo_2023-03-31-10-14-57/franka_state_controller-O_T_EE.csv',
        curr_path + '/config/Data_Trajs/demo_2023-03-31-10-16-49/franka_state_controller-O_T_EE.csv']
    start_indx_all = np.array([550, 300])
    end_indx_all = np.array([1150, 800])  # substract from end time
    model_file_name = curr_path + '/config/Wiping_franka_LR_no_norm_no_t_checkpoint.eqx'
    ## TB
    # csv_files_all = [
    #     '/home/farhadnawaz/catkin_ws/src/franka_interactive_controllers/config/Data_Trajs/demo_2023-03-31-10-18-42/franka_state_controller-O_T_EE.csv',
    #     '/home/farhadnawaz/catkin_ws/src/franka_interactive_controllers/config/Data_Trajs/demo_2023-03-31-10-19-07/franka_state_controller-O_T_EE.csv',
    #     '/home/farhadnawaz/catkin_ws/src/franka_interactive_controllers/config/Data_Trajs/demo_2023-03-31-10-19-30/franka_state_controller-O_T_EE.csv']
    # start_indx_all = np.array([400, 200, 300])
    # end_indx_all = np.array([300, 600, 200])  # substract from end time

    traj_all_process, traj_vel_all, time, scaler_all_t = read_data_csv(csv_files_all, start_indx_all, end_indx_all)    
    model_load = load_model(traj_all_process, time, model_file_name, d_flag=0)
    rospy.init_node('desired_vel', anonymous=True)
    rate = rospy.Rate(1000)  # 1000hz

    desired_vel_obj = desired_vel(traj_all_process, time, scaler_all_t, model_load)

    while not rospy.is_shutdown():
        desired_vel_obj.cmd_desired_vel()
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
