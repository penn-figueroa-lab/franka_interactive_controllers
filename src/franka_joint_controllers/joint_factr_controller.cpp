// This code was derived from franka_example controllers
// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
// Current development and modification of this code by Nadia Figueroa (MIT) 2021.

#include <joint_factr_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <pseudo_inversion.h>
#include <hardware_interface/joint_command_interface.h>


//std_msgs/Float32MultiArray

namespace franka_interactive_controllers {

bool JointFactrController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {

  sub_control_signal = node_handle.subscribe("/joint_factr_controller/desired_joint_pos", 1000, &JointFactrController::controller_callback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  pub_ft = node_handle.advertise<geometry_msgs::WrenchStamped>("/franka_ft", 10);
  dq_prev.setZero();

  pub_trq = node_handle.advertise<std_msgs::Float32MultiArray>("/joint_factr_controller/external_torque", 10);

  // Getting ROSParams
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("JointFactrController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "JointFactrController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  // Initialize variables for tool compensation from yaml config file
  activate_tool_compensation_ = true;
  std::vector<double> external_tool_compensation;
  if (!node_handle.getParam("external_tool_compensation", external_tool_compensation)) {
      ROS_ERROR(
          "JointFactrController: Invalid or no external_tool_compensation parameters provided, "
          "aborting controller init!");
      return false;
    }

  tool_compensation_force_.setZero();  
  for (size_t i = 0; i < 6; ++i) 
    tool_compensation_force_[i] = external_tool_compensation.at(i);
  ROS_INFO_STREAM("External tool compensation force: " << std::endl << tool_compensation_force_);
  // tool_compensation_force_ << 0.46, -0.17, -1.64, 0, 0, 0;  //read from yaml

  // Getting libranka control interfaces
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointFactrController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointFactrController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointFactrController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointFactrController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointFactrController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointFactrController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  // Getting Dynamic Reconfigure objects
  dynamic_reconfigure_gravity_compensation_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "dynamic_reconfigure_gravity_compensation_param_node");

  dynamic_server_gravity_compensation_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_interactive_controllers::gravity_compensation_paramConfig>>(

      dynamic_reconfigure_gravity_compensation_param_node_);
  dynamic_server_gravity_compensation_param_->setCallback(
      boost::bind(&JointFactrController::gravitycompensationParamCallback, this, _1, _2));

  
  // Initialize variables for joint locks
  set_locked_joints_position_ = false;
  activate_lock_joint6_ = false;
  activate_lock_joint7_ = false;
  k_lock_      = 50; 
  q_locked_joints_.setZero();



  q_desired = Eigen::VectorXd::Zero(7);
  received_command = false;


  return true;
}

void JointFactrController::starting(const ros::Time& /*time*/) {
  // Get robot current/initial joint state
  franka::RobotState initial_state = state_handle_->getRobotState();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());

  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);


  std::array<double, 7> q_start{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
  for (size_t i = 0; i < q_start.size(); i++) {
    if (std::abs(q_initial[i] - q_start[i]) > 0.1) {
      ROS_ERROR_STREAM(
          "JointFactrController: Robot is not in the expected starting position for "
          "running this example.");
    }
  }
}

void JointFactrController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& /*period*/) {
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  std::array<double, 49> mass_array = model_handle_->getMass();
  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 7>> mass(mass_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());

  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Map<Eigen::Matrix<double, 4,4>> end_T(robot_state.O_T_EE.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_est(robot_state.tau_ext_hat_filtered.data());

  
  Eigen::VectorXd tau_dyn(7), tau_contact(7), wrench_contact_K(6), dq_filt(7), ddq(7);
  for (auto i=0; i< 7; i++){
    dq_filt(i) = lpf[i].filt(dq(i));
  }

  ddq = (dq_filt - dq_prev) / 0.001;
  dq_prev = dq_filt;

  tau_dyn = coriolis + mass * ddq;
  tau_contact = tau_est - tau_dyn;
  wrench_contact_K = jacobian.transpose().completeOrthogonalDecomposition().solve(tau_contact);
  
  geometry_msgs::WrenchStamped wrench_msg;
  wrench_msg.header.stamp = ros::Time::now();
  wrench_msg.header.frame_id = "panda_K";
  wrench_msg.wrench.force.x = wrench_contact_K(0);
  wrench_msg.wrench.force.y = wrench_contact_K(1);
  wrench_msg.wrench.force.z = wrench_contact_K(2);

  pub_ft.publish(wrench_msg);


  std_msgs::Float32MultiArray ext_torque_msg;
  ext_torque_msg.data.resize(7);
  for (size_t i = 0; i < 7; ++i) {
  ext_torque_msg.data[i] = static_cast<float>(tau_est[i]);
  }
  pub_trq.publish(ext_torque_msg);




  /////////////////////////////////////////////////////////////////////
  // allocate variables
  Eigen::VectorXd tau_d(7), tau_task(7), tau_nullspace(7), tau_tool(7);
  Eigen::Matrix<double, 7, 1> K;
  Eigen::Matrix<double, 7, 1> D;
  K << 70, 70, 70, 70, 70, 70, 70;
  D << 50, 50, 50, 25, 15, 10, 5; 

  // // pseudoinverse for nullspace handling kinematic pseudoinverse
  // Eigen::MatrixXd jacobian_transpose_pinv;
  // pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // Set 0 torques for the controller
  // tau_task.setZero();

  // 
  // Desired torque (Check this.. might not be necessary)
  //tau_d << tau_task + coriolis - tau_tool;
  // printf("%f\n", tau_received[4]);
  // std::cout << "I am re" << std::endl;

  Eigen::Matrix<double, 7, 1> dq_desired;
  dq_desired.setZero();

  double max_allowed_error = 0.3; // radians, adjust as needed

  Eigen::Matrix<double, 7, 1> position_error = q_desired - q;

  if (!received_command) {
    std::cout << "No command" << std::endl;
    tau_d.setZero();
  }else if (position_error.cwiseAbs().maxCoeff() > max_allowed_error){
    std::cout << "Exceeding max error: "<<  position_error << std::endl;
    tau_d.setZero();
  }else{
    tau_d << K.cwiseProduct(q_desired - q) + D.cwiseProduct(dq_desired - dq);
  }

  // tau_d << tau_received;

  // Alternative 
  // tau_d.setZero();
  //std::cout << "send torque" << std::endl;

  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRate(tau_d, tau_J_d);

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));
  }
}

Eigen::Matrix<double, 7, 1> JointFactrController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void JointFactrController::gravitycompensationParamCallback(
    franka_interactive_controllers::gravity_compensation_paramConfig& config,
    uint32_t /*level*/) {
  
  // To activate external tool compensation
  activate_tool_compensation_ = config.activate_tool_compensation;
  
  // To lock a specific joint
  activate_lock_joint6_                = config.activate_lock_joint6;
  activate_lock_joint7_                = config.activate_lock_joint7;
  
  set_locked_joints_position_ = config.set_locked_joints_position;
  if (set_locked_joints_position_){
      franka::RobotState locked_state = state_handle_->getRobotState();
      Eigen::Map<Eigen::Matrix<double, 7, 1>> q_locked_joints(locked_state.q.data());
      q_locked_joints_ = q_locked_joints;
      ROS_INFO_STREAM("Locked Joints Set to: " << q_locked_joints_);
  }
}

void JointFactrController::controller_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
  // std::cout << "I am re" << std::endl;
  for(int i = 0; i<7; i++)
  {
      q_desired[i] = msg->data[i];
  }
  // printf("%f\n", tau_received[4]);

  received_command = true;
}


}  // namespace franka_interactive_controllers

PLUGINLIB_EXPORT_CLASS(franka_interactive_controllers::JointFactrController,
                       controller_interface::ControllerBase)
