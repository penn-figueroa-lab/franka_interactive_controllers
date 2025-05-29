// This code was derived from franka_example controllers
// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
// Current development and modification of this code by Nadia Figueroa (MIT) 2021.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <franka_interactive_controllers/gravity_compensation_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>

namespace franka_interactive_controllers {

class LPF{
    public:
        explicit LPF(double a=0.1): a_(a), first_(true), y_(0) {}
        double filt(double x){ if(first_){y_=x; first_=false;} y_+=a_*(x-y_); return y_;}
    private: double a_, y_; bool first_;
};

class JointGravityCompensationController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  const double delta_tau_max_{1.0};

  // Variables for tool compensation
  bool activate_tool_compensation_;
  Eigen::Matrix<double, 6, 1> tool_compensation_force_;


  // Variables for joint locks
  Eigen::Matrix<double, 7, 1> q_locked_joints_;
  bool activate_lock_joint6_;
  bool activate_lock_joint7_;
  bool set_locked_joints_position_;
  double k_lock_;
  

  // Dynamic reconfigure
  std::unique_ptr<dynamic_reconfigure::Server<franka_interactive_controllers::gravity_compensation_paramConfig>>
      dynamic_server_gravity_compensation_param_;
  ros::NodeHandle dynamic_reconfigure_gravity_compensation_param_node_;
  void gravitycompensationParamCallback(franka_interactive_controllers::gravity_compensation_paramConfig& config,
                               uint32_t level);
  void controller_callback(const std_msgs::Float32MultiArray::ConstPtr& msg);

  ros::Publisher pub_state;
  ros::Publisher pub_try;
  ros::Publisher pub_ft;
  ros::Subscriber sub_control_signal;
  Eigen::VectorXd tau_received;


  std::array<LPF, 7> lpf;
  Eigen::Matrix<double, 7, 1> dq_prev;

};

}  // namespace franka_interactive_controllers
