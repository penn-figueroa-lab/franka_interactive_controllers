// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>
#include <std_msgs/String.h>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

// #include "controllers_common.h"
// #include <franka_motion_generators/libfranka_joint_motion_generator.h>
#include <libfranka_joint_motion_generator.h>

#include "ros/ros.h"
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/JointState.h>

namespace {
template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}  // anonymous namespace

std::string latest_command = "none";
void commandCallback(const std_msgs::String::ConstPtr& msg) {
  latest_command = msg->data;
}

std::mutex mtx;
std::array<double, 7> latest_q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
float latest_gripper_width = 0.0;
bool new_goal_available = false;

void goalCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
  // if (msg->data.size() != 8) {
  //   ROS_WARN("Received joint goal with size %lu, expected 8", msg->data.size());
  //   return;
  // }

  std::lock_guard<std::mutex> lock(mtx);
  for (size_t i = 0; i < 7; ++i) {
    latest_q_goal[i] = msg->data[i];
  }
  // latest_gripper_width = msg->data[7];
  new_goal_available = true;
}

int main(int argc, char** argv) {

  ros::init(argc, argv, "franka_joint_goal_motion_generator_node");
  ros::NodeHandle nh;
  ros::Subscriber command_sub = nh.subscribe("string_command_topic", 10, commandCallback);
  ros::Subscriber goal_sub = nh.subscribe("/pizero/action", 1, goalCallback);

  ros::Publisher joint_state_pub = nh.advertise<sensor_msgs::JointState>("/franka_state_controller/joint_states", 10);


  // ros::NodeHandle _nh("~");
  std::string franka_ip = "172.16.0.2";



  try {
    // Connect to robot.
    franka::Robot robot(franka_ip);
    franka::Gripper gripper(franka_ip);

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}});

  
    while (ros::ok()) {
      ros::spinOnce();

      // Publish joint state
      franka::RobotState state = robot.readOnce();

      sensor_msgs::JointState joint_msg;
      joint_msg.header.stamp = ros::Time::now();
      joint_msg.name = {
        "panda_joint1", "panda_joint2", "panda_joint3", 
        "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
      };
      joint_msg.position = std::vector<double>(state.q.begin(), state.q.end());
      joint_msg.velocity = std::vector<double>(state.dq.begin(), state.dq.end());
      joint_msg.effort   = std::vector<double>(state.tau_J.begin(), state.tau_J.end());
      joint_state_pub.publish(joint_msg);
    
      std::array<double, 7> current_goal;
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (!new_goal_available) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        current_goal = latest_q_goal;
        new_goal_available = false;
      }
      std::cout << "AFTER Latest string command: " << latest_command << std::endl;
      
    
      std::cout << "Received and executing joint goal: ";
      for (size_t i = 0; i < current_goal.size(); ++i) {
        std::cout << current_goal[i] << (i < current_goal.size() - 1 ? ", " : "\n");
      }
      std::cout << "gripper width: " << latest_gripper_width << std::endl;
      MotionGenerator motion_generator(0.5, current_goal);
      robot.control(motion_generator);
      bool success = gripper.move(latest_gripper_width, 0.1);
    }

  } catch (const franka::Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }

  // Stop the node's resources
  // ros::shutdown();
  // Exit tranquilly
  return 0;
}
