#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string>
#include <thread>

#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include <unitree/common/thread/thread.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>

#include "Joystick.hpp"
#include "Interface.hpp"
#include "Types.h"
#include "fort.hpp"
#include "logger.hpp"
#include "motors.hpp"

#include "base_state.h"
#include "data_buffer.hpp"
#include "motors.hpp"

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

#define USE_JOYSTICK true

#define STATUS_INIT 0
#define STATUS_WAITING_AIR 1
#define STATUS_GAIN_TRANSITION 2
#define STATUS_RUN 3
#define STATUS_DAMPING 4

static const std::string kTopicLowCommand = "rt/lowcmd";
static const std::string kTopicLowState = "rt/lowstate";

class HumanoidExample;
void waiting(HumanoidExample *HE);

class HumanoidExample {
public:
  HumanoidExample(const std::string &networkInterface = "",
                  const std::string &model_file = "")
      : networkInterface_() {

    // unitree::robot::ChannelFactory::Instance()->Init(1, "lo");
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);
    std::cout << "Initialize channel factory." << std::endl;

    msc.reset(new unitree::robot::b2::MotionSwitcherClient());
    msc->SetTimeout(2.0F);
    msc->Init();

    /*Shut down  motion control-related service*/
    while(queryMotionStatus())
    {
        std::cout << "Try to deactivate the motion control-related service." << std::endl;
        int32_t ret = msc->ReleaseMode(); 
        if (ret == 0) {
            std::cout << "ReleaseMode succeeded." << std::endl;
        } else {
            std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
        }
        sleep(5);
    }

    lowcmd_publisher_.reset(
        new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(
            kTopicLowCommand));
    lowcmd_publisher_->InitChannel();
    command_writer_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "command_writer", UT_CPU_ID_NONE, 2000,
        &HumanoidExample::LowCommandWriter, this);

    lowstate_subscriber_.reset(
        new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(
            kTopicLowState));
    lowstate_subscriber_->InitChannel(
        std::bind(&HumanoidExample::LowStateHandler, this,
                  std::placeholders::_1),
        1);
    int control_period_us = control_dt_ * 1e6;
    control_thread_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "control", UT_CPU_ID_NONE, control_period_us, &HumanoidExample::Control,
        this);

    int report_period_us = report_dt_ * 1e6;
    report_sensors_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "report_sensors", UT_CPU_ID_NONE, report_period_us,
        &HumanoidExample::UpdateTables, this, false);

    // Scale the policy control gains
    // kp_ *= 0.0;
    // kd_ *= 0.0;
    // kp_wait_ *= 0.0;
    // kd_wait_ *= 0.0;
  
    // Create the link with the joystick
    if (USE_JOYSTICK) {
      joy_.initialize(control_dt_);
    }

    int joystick_period_us = 0.001 * 1e6;
    joystick_thread_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "joystick", UT_CPU_ID_NONE, joystick_period_us, &HumanoidExample::ReadJoystick,
        this);

    int logging_period_us = 0.002 * 1e6;
    logging_thread_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "logging", UT_CPU_ID_NONE, logging_period_us, &HumanoidExample::LogAll,
        this);

    // Create link with network interface
    networkInterface_.initialize(model_file, q_init_.head(19), control_dt_);
    policy_out_ = Vxf::Zero(networkInterface_.get_actDim());

    // Initialize tables for console display
    UpdateTables(true);

    // Initialize sink for data logging
    fmtlog::setHeaderPattern("");
    fmtlog::setLogFile(getCurrentDateTime());
    fmtlog::setFlushDelay(100000000);
    fmtlog::startPollingThread(100000000);
  }

  // Default destructor
  ~HumanoidExample() = default;

  // Get the current date in local time
  char *getCurrentDateTime();

  // Prepare the command message and send it to the publisher
  void LowCommandWriter();

  // Update motor and base states using received sensor message
  void LowStateHandler(const void *message);

  // Update motor and base states using received sensor messagevoid LowStateHandler(const void *message);
  void ReadJoystick(); // {joy_.update_v_ref(0);}

  // Main control function
  void Control();

  // Basic print of sensor data to the console
  void ReportSensors();

  // Launch controller once Enter is pressed
  void endWaiting();

private:
  void RecordMotorState(const unitree_go::msg::dds_::LowState_ &msg);
  void RecordBaseState(const unitree_go::msg::dds_::LowState_ &msg);

  // Refresh the quantities in the tables displayed in the console
  void UpdateTables(bool init = false);

  // Log all monitored quantities for the current time step
  void LogAll();

  // Check if a motor index corresponds to a "weak" motor
  inline bool IsWeakMotor(int motor_index) {
    return motor_index == JointIndex::kLeftAnkle ||
           motor_index == JointIndex::kRightAnkle ||
           motor_index == JointIndex::kRightShoulderPitch ||
           motor_index == JointIndex::kRightShoulderRoll ||
           motor_index == JointIndex::kRightShoulderYaw ||
           motor_index == JointIndex::kRightElbow ||
           motor_index == JointIndex::kLeftShoulderPitch ||
           motor_index == JointIndex::kLeftShoulderRoll ||
           motor_index == JointIndex::kLeftShoulderYaw ||
           motor_index == JointIndex::kLeftElbow;
  }

  std::string queryServiceName(std::string form,std::string name)
  {
      if(form == "0")
      {
          if(name == "normal" ) return "sport_mode"; 
          if(name == "ai" ) return "ai_sport"; 
          if(name == "advanced" ) return "advanced_sport"; 
      }
      else
      {
          if(name == "ai-w" ) return "wheeled_sport(go2W)"; 
          if(name == "normal-w" ) return "wheeled_sport(b2W)";
      }
      return "";
  }

  int queryMotionStatus()
  {
      std::string robotForm,motionName;
      int motionStatus;
      int32_t ret = msc->CheckMode(robotForm,motionName);
      if (ret == 0) {
          std::cout << "CheckMode succeeded." << std::endl;
      } else {
          std::cout << "CheckMode failed. Error code: " << ret << std::endl;
      }
      if(motionName.empty())
      {
          std::cout << "The motion control-related service is deactivated." << std::endl;
          motionStatus = 0;
      }
      else
      {
          std::string serviceName = queryServiceName(robotForm,motionName);
          std::cout << "Service: "<< serviceName<< " is activate" << std::endl;
          motionStatus = 1;
      }
      return motionStatus;
  }

  unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>
      lowcmd_publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>
      lowstate_subscriber_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<BaseState> base_state_buffer_;

  std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc;

  // control params
  const float control_dt_ = 0.01f;
  const float init_duration_ = 5.f;
  const float interp_duration_ = 0.1f;
  const float report_dt_ = 0.1f;

  int status_ = STATUS_INIT;

  float hip_pitch_init_pos_ = -0.5f;
  float knee_init_pos_ = 1.f;
  float ankle_init_pos_ = -0.5f;
  float shoulder_pitch_init_pos_ = 0.4f;

  float time_ = 0.f;
  float time_run_ = 0.f;
  float time_log_ = 0.f;

  // Default configuration
  const Vector20 q_init_{
      0.0, 0.0, -0.2, 0.6, -0.4, 0.0, 0.0, -0.2,  0.6, -0.4, // Legs
      0.0, 0.4,  0.0, 0.0, -0.4, 0.4, 0.0,  0.0, -0.4,       // Torso and arms
      0.0};                                                  // Unused joint
  const Vector20 q_lim_lower{-0.43, -0.43, -3.14, -0.26, -0.87,
                             -0.43, -0.43, -3.14, -0.26, -0.87, // Legs
                             -2.35, -2.87, -0.34, -1.3,  -1.25,
                             -2.87, -3.11, -4.45, -1.25, // Torso and arms
                             0.0};                       // Unused joint
  const Vector20 q_lim_upper{
      0.43, 0.43, 2.53, 2.05, 0.52,
      0.43, 0.43, 2.53, 2.05, 0.52, // Legs
      2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3,  2.61, // Torso and arms
      0.0};                                                 // Unused joint

  // Proportional derivative gains
  Vector20 kp_{200.0, 200.0, 200.0, 300.0, 40.0,
               200.0, 200.0, 200.0, 300.0, 40.0, // Legs
               300.0, 40.0, 40.0, 40.0, 40.0,
               40.0, 40.0, 40.0, 40.0, // Torso and arms
               0.0};                   // Unused joint

  Vector20 kd_{5.0, 5.0, 5.0, 6.0, 2.0, 5.0, 5.0, 5.0, 6.0, 2.0, // Legs
               6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // Torso and arms
              0.0};

  Vector20 kp_wait_{1500.0, 1500.0, 1500.0, 1500.0, 1500.0,
                    1500.0, 1500.0, 1500.0, 1500.0, 1500.0, // Legs
                    200.0,  200.0,  100.0,  100.0,  200.0,
                    200.0,  100.0,  100.0,  200.0, // Torso and arms
                    0.0};                          // Unused joint

  Vector20 kd_wait_{
      25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, // Legs
      6.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0, // Torso and arms
      0.0};                                                // Unused joint

  std::array<float, kNumMotors> desired_torques_ = {};
  std::array<float, 19> policy_log_ = {};

  Vector6 cmd_ = Vector6::Zero();

  Vxf policy_out_;

  // Network interface
  Interface networkInterface_;

  // Joystick interface
  Joystick joy_;

  // multithreading
  unitree::common::ThreadPtr command_writer_ptr_;
  unitree::common::ThreadPtr control_thread_ptr_;
  unitree::common::ThreadPtr report_sensors_ptr_;
  unitree::common::ThreadPtr joystick_thread_ptr_;
  unitree::common::ThreadPtr logging_thread_ptr_;

  // Tables for console display
  fort::char_table table_IMU_;
  fort::char_table table_legs_;
  fort::char_table table_arms_;
  fort::char_table table_misc_;

  // Reordering quaternion vector
  const Eigen::Matrix<float, 4, 4> quatPermut{{0, 1, 0, 0},
                                              {0, 0, 1, 0},
                                              {0, 0, 0, 1},
                                              {1, 0, 0, 0}};

};

////
// WAITING BEFORE LAUNCHING CONTROLLER
////

// Wait for Enter key press
void waiting(HumanoidExample *HE) {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(1000ms);
  std::cin.get();
  HE->endWaiting();
}

void HumanoidExample::endWaiting() {
  if (status_ == STATUS_WAITING_AIR) {
    time_run_ = -control_dt_;
    status_ = STATUS_GAIN_TRANSITION;
  }
}

////
// Reading info from robot and writing commands to robot
////

void HumanoidExample::LowCommandWriter() {
  unitree_go::msg::dds_::LowCmd_ dds_low_command{};
  dds_low_command.head()[0] = 0xFE;
  dds_low_command.head()[1] = 0xEF;
  dds_low_command.level_flag() = 0xFF;
  dds_low_command.gpio() = 0;

  const std::shared_ptr<const MotorCommand> mc_tmp_ptr =
      motor_command_buffer_.GetData();
  if (mc_tmp_ptr) {
    for (int i = 0; i < kNumMotors; ++i) {
      if (IsWeakMotor(i)) {
        dds_low_command.motor_cmd().at(i).mode() = (0x01);
      } else {
        dds_low_command.motor_cmd().at(i).mode() = (0x0A);
      }
      dds_low_command.motor_cmd().at(i).tau() = mc_tmp_ptr->tau_ff.at(i);
      dds_low_command.motor_cmd().at(i).q() = mc_tmp_ptr->q_ref.at(i);
      dds_low_command.motor_cmd().at(i).dq() = mc_tmp_ptr->dq_ref.at(i);
      dds_low_command.motor_cmd().at(i).kp() = mc_tmp_ptr->kp.at(i);
      dds_low_command.motor_cmd().at(i).kd() = mc_tmp_ptr->kd.at(i);
    }
    dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command,
                                      (sizeof(dds_low_command) >> 2) - 1);
    lowcmd_publisher_->Write(dds_low_command);
  }
}

void HumanoidExample::LowStateHandler(const void *message) {
  unitree_go::msg::dds_::LowState_ low_state =
      *(unitree_go::msg::dds_::LowState_ *)message;

  RecordMotorState(low_state);
  RecordBaseState(low_state);
}

void HumanoidExample::RecordMotorState(const unitree_go::msg::dds_::LowState_ &msg) {
  MotorState ms_tmp;
  for (int i = 0; i < kNumMotors; ++i) {
    ms_tmp.q.at(i) = msg.motor_state()[i].q();
    ms_tmp.dq.at(i) = msg.motor_state()[i].dq();
    ms_tmp.tau.at(i) = msg.motor_state()[i].tau_est();
  }

  motor_state_buffer_.SetData(ms_tmp);
}

void HumanoidExample::RecordBaseState(const unitree_go::msg::dds_::LowState_ &msg) {
  BaseState bs_tmp;
  bs_tmp.omega = msg.imu_state().gyroscope();
  bs_tmp.quat = msg.imu_state().quaternion();
  bs_tmp.rpy = msg.imu_state().rpy();
  bs_tmp.acc = msg.imu_state().accelerometer();

  base_state_buffer_.SetData(bs_tmp);
}

void HumanoidExample::ReadJoystick() {
  joy_.update_v_ref(0);
}

////
// MAIN CONTROL FUNCTION
////

void HumanoidExample::Control() {
  MotorCommand motor_command_tmp;
  const std::shared_ptr<const MotorState> ms_tmp_ptr =
      motor_state_buffer_.GetData();
  const std::shared_ptr<const BaseState> bs_tmp_ptr =
      base_state_buffer_.GetData();

  if (ms_tmp_ptr && bs_tmp_ptr) {
    time_ += control_dt_;

    Vector20 pos, vel, tau;
    for (int i = 0; i < kNumMotors; ++i) {
      pos(i) = ms_tmp_ptr->q.at(moti[i]);
      vel(i) = ms_tmp_ptr->dq.at(moti[i]);
      tau(i) = ms_tmp_ptr->tau.at(moti[i]);
    }

    // Check if joints are too close from position limits
    const bool lim_lower = ((pos - 0.95 * q_lim_lower).array() < 0.0).any();
    const bool lim_upper = ((pos - 0.95 * q_lim_upper).array() > 0.0).any();
    if (lim_lower || lim_upper) {
      std::cout << "Pos threshold breached!!!" << std::endl;
      std::cout << "UP : " << std::fixed << std::setprecision(4) << (0.95 * q_lim_upper).transpose() << std::endl;
      std::cout << "POS: " << std::fixed << std::setprecision(4) << pos.transpose() << std::endl;
      std::cout << "LOW: " << std::fixed << std::setprecision(4) << (0.95 * q_lim_lower).transpose() << std::endl;
      // status_ = STATUS_DAMPING;
    }

    // Check if joint velocities are too high
    const bool lim_velocity = ((vel.array().abs() - 12) > 0.0).any();
    if (lim_velocity) {
      std::cout << "Velocity threshold breached!!!" << std::endl;
      std::cout << "VEL: " << std::fixed << std::setprecision(4) << vel.transpose() << std::endl;
      // status_ = STATUS_DAMPING;
    }

    // Switch to waiting after initialization
    if ((status_ == STATUS_INIT) && (time_ > init_duration_)) {
      status_ = STATUS_WAITING_AIR;
      std::thread wait_thread(waiting, this);
      wait_thread.detach();
    }

    switch (status_) {
    case STATUS_RUN: {
      time_run_ += control_dt_;

      /*
      // Interpolation coefficient to slowly switch PD gains
      float alpha = 1.0;
      if (time_run_ < interp_duration_) {
        alpha = time_run_ / interp_duration_;
      }

      for (int i = 0; i < kNumMotors; ++i) {
        motor_command_tmp.kp.at(moti[i]) = kp_wait_(i) * (1 - alpha) + kp_(i) * alpha;
        motor_command_tmp.kd.at(moti[i]) = kd_wait_(i) * (1 - alpha) + kd_(i) * alpha;
        motor_command_tmp.q_ref.at(moti[i]) = q_init_(i);
        motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
        motor_command_tmp.tau_ff.at(moti[i]) = 0.f;
      }
      // Inference to get position targets from the policy
      policy_out_ = networkInterface_.forward();
      for (int i = 0; i < 10; ++i) {
        policy_log_[i] = policy_out_[i];
      }

      if (time_run_ < interp_duration_) {
        break;
      }
      */

      // Refresh joystick
      if (USE_JOYSTICK) {
        cmd_ = joy_.getVRef();

        for (int i = 0; i < 6; ++i) {
          if (std::abs(cmd_(i)) < 0.1) {cmd_(i) = 0.0;}
        }

        //std::cout << cmd_.transpose() << std::endl;
        // cmd_ = Vector6::Zero();
        // cmd_(0) = 0.4;

        /*cmd_ = Vector6::Zero();
        cmd_(0) = std::min(0.4, 0.4 * time_run_ / 1.0);
        // cmd_(5) = 0.0;
        if (time_run_ > 5.0) {cmd_(0) = 0.0; cmd_(5) = 0.0;}*/

        if (joy_.getCross()) {status_ = STATUS_DAMPING;}

      } else {
        cmd_ = Vector6::Zero();
        cmd_(0) = 0.0;
      }

      /*
      time_nonzero += control_dt_;
      for (int i = 0; i < 6; ++i) {
        if (std::abs(cmd_(i)) > 0.1) {time_nonzero = 0.0;}
      }

      // Get [-0.5, 0.5] modulo of gait cycle
      float phase = std::fmod(1.2 * time_run_, 1.0);
      if (phase > 0.5) {phase -= 1.0;}

      // Change gait mode during double support phase
      if (-0.1 < phase && phase < 0.1) {
        if (time_nonzero == 0.0) {loco_mode = 0.0;}
        if (time_nonzero > 2.0) {loco_mode = 1.0;}
      }
      */

      //std::cout << "LOCO " << loco_mode << std::endl;

      Vector3 rpy(bs_tmp_ptr->rpy.data());
      Vector4 ori(bs_tmp_ptr->quat.data());
      Vector3 gyro(bs_tmp_ptr->omega.data());

      // std::cout << rpy.transpose() << std::endl;

      // Update observation vector (ManiSkill)
      networkInterface_.update_observation_ManiSkill(pos.head(19), vel.head(19), tau.head(19), rpy,
                                                  quatPermut * ori, gyro, cmd_, time_run_);

      // Inference to get position targets from the policy (ManiSkill)
      policy_out_ = networkInterface_.forward_ManiSkill();

      /*
      // Update observation vector (Mujoco)
      networkInterface_.update_observation_with_clock(pos.head(19), vel.head(19), tau.head(19), rpy,
                                                  quatPermut * ori,  gyro, cmd_, 0.5 + time_run_);
      
      // Inference to get position targets from the policy (Mujoco)
      policy_out_ = networkInterface_.forward();
      */

      /*
      // DEBUG: Apply sinusoidal torque command to a given joint of both legs
      policy_out_ = Vxf::Zero(19);
      policy_out_ += q_init_.head(19);
      float freq = 1.0; //std::floor(time_run_ / 2) + 1;
      float tgt = 2.0 * std::sin(2 * pi_v * freq * time_run_);
      const int Ni = 3;
      policy_out_(Ni) = policy_out_(Ni);// + tgt;
      policy_out_(Ni + 5) = policy_out_(Ni + 5);// - tgt;

      if (time_run_ > 6.0) {
        status_ = STATUS_DAMPING;
      }
      */

      // Check policy output size
      assert(policy_out_.rows() == 19);

      // Logging policy output
      for (int i = 0; i < policy_out_.rows(); ++i) {
        policy_log_[i] = policy_out_[i];
      }

      //std::cout << policy_out_.transpose() << std::endl;
      /* std::cout << policy_out_.rows() << std::endl;*/
      //std::cout << "ActDim: " << networkInterface_.get_actDim() << std::endl;

      // Send policy commands to the robot
      Vxf network_cmd = policy_out_;

      /*
      network_cmd = q_init_;
      float pulse = std::fmod(time_run_, 2.0);
      float offset = 0.0;
      if (pulse > 1.0) {offset = -0.2;}
      network_cmd(1) = q_init_(1) + offset;
      network_cmd(6) = q_init_(6) + offset;
      */

      float q_des = 0.f;
      for (int i = 0; i < kNumMotors; ++i) {
        q_des = i < networkInterface_.get_actDim() ? network_cmd(i) : q_init_(i);
        motor_command_tmp.kp.at(moti[i]) = kp_(i);
        motor_command_tmp.kd.at(moti[i]) = kd_(i);
        motor_command_tmp.q_ref.at(moti[i]) = q_des;
        motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
        motor_command_tmp.tau_ff.at(moti[i]) = 0.f; // ((i == 3) || (i == 8)) ? tgt : 0.0;
      }
      break;
    }
    case STATUS_WAITING_AIR: {
      // Wait at default configuration
      for (int i = 0; i < kNumMotors; ++i) {
        motor_command_tmp.kp.at(moti[i]) = kp_wait_(i);
        motor_command_tmp.kd.at(moti[i]) = kd_wait_(i);
        motor_command_tmp.q_ref.at(moti[i]) = q_init_(i);
        motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
        motor_command_tmp.tau_ff.at(moti[i]) = 0.f;
      }
      break;
    }
    case STATUS_GAIN_TRANSITION: {

      bool start = true;
      if (USE_JOYSTICK) {
        start = (joy_.getStart()==1);
      }

      // Interpolation from waiting gains to policy gains
      float alpha = 0;
      if (start) {
        time_run_ += control_dt_;
        alpha = std::clamp(0.f, 1.f, time_run_ / interp_duration_);
      }

      // Slowly switch PD gains to policy gains
      for (int i = 0; i < kNumMotors; ++i) {
        motor_command_tmp.kp.at(moti[i]) = kp_wait_(i) * (1 - alpha) + kp_(i) * alpha;
        motor_command_tmp.kd.at(moti[i]) = kd_wait_(i) * (1 - alpha) + kd_(i) * alpha;
        motor_command_tmp.q_ref.at(moti[i]) = q_init_(i);
        motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
        motor_command_tmp.tau_ff.at(moti[i]) = 0.f;
      }

      // If transition is over, switch to the policy
      if (time_run_ >= interp_duration_ && start) {
        time_run_ = -control_dt_;
        status_ = STATUS_RUN;
      }
      break;
    }
    case STATUS_INIT: {
      // Slowly move to default configuration
      float ratio = std::clamp(time_, 0.f, init_duration_) / init_duration_;
      for (int i = 0; i < kNumMotors; ++i) {
        motor_command_tmp.kp.at(moti[i]) = kp_wait_(i);
        motor_command_tmp.kd.at(moti[i]) = kd_wait_(i);
        motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
        motor_command_tmp.tau_ff.at(moti[i]) = 0.f;

        float q_des = (q_init_(i) - ms_tmp_ptr->q.at(moti[i])) * ratio +
                      ms_tmp_ptr->q.at(moti[i]);
        motor_command_tmp.q_ref.at(moti[i]) = q_des;
      }
      break;
    }
    default: { // case STATUS_DAMPING:
      // Emergency damping, no Kp, only Kd with 0 ref vel
      for (int i = 0; i < kNumMotors; ++i) {
        motor_command_tmp.kp.at(moti[i]) = 0.f;
        motor_command_tmp.kd.at(moti[i]) = kd_(i);
        motor_command_tmp.q_ref.at(moti[i]) = ms_tmp_ptr->q.at(moti[i]);
        motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
        motor_command_tmp.tau_ff.at(moti[i]) = 0.f;
      }
    }
    }
    // Write to command buffer
    motor_command_buffer_.SetData(motor_command_tmp);

    // Log sensors and commands
    for (int i = 0; i < kNumMotors; ++i) {
      desired_torques_[i] =
          motor_command_tmp.kp.at(moti[i]) *
              (motor_command_tmp.q_ref.at(moti[i]) - ms_tmp_ptr->q.at(moti[i])) +
          motor_command_tmp.kd.at(moti[i]) *
              (motor_command_tmp.dq_ref.at(moti[i]) - ms_tmp_ptr->dq.at(moti[i])) +
          motor_command_tmp.tau_ff.at(moti[i]);
    }
    // LogAll();
  }
}

////
// LOGGING DURING EXPERIMENT
////

void HumanoidExample::LogAll() {

  // Retrieve and store data
  const std::shared_ptr<const MotorState> ms_tmp_ptr =
      motor_state_buffer_.GetData();
  const std::shared_ptr<const MotorCommand> mc_tmp_ptr =
      motor_command_buffer_.GetData();
  const std::shared_ptr<const BaseState> bs_tmp_ptr =
      base_state_buffer_.GetData();

  time_log_ += 0.002;

  // Log all monitored variables
  logi("time,{}", time_log_);
  if (ms_tmp_ptr) {
    logi("{}", *ms_tmp_ptr);
  }
  if (mc_tmp_ptr) {
    logi("{}", *mc_tmp_ptr);
  }
  if (bs_tmp_ptr) {
    logi("{}", *bs_tmp_ptr);
  }
  logi("{}", "tau_des," + arrayToStringView(desired_torques_));
  logi("{}", "policy_out," + arrayToStringView(policy_log_));
}

////
// DISPLAY IN CONSOLE
////

void HumanoidExample::ReportSensors() {
  const std::shared_ptr<const BaseState> bs_tmp_ptr =
      base_state_buffer_.GetData();
  const std::shared_ptr<const MotorState> ms_tmp_ptr =
      motor_state_buffer_.GetData();
  if (bs_tmp_ptr) {
    // Roll Pitch Yaw orientation
    std::cout << std::setprecision(4) << "rpy: [" << bs_tmp_ptr->rpy.at(0)
              << ", " << bs_tmp_ptr->rpy.at(1) << ", " << bs_tmp_ptr->rpy.at(2)
              << "]" << std::endl;
    // Gyroscope
    std::cout << std::setprecision(4) << "gyro: [" << bs_tmp_ptr->omega.at(0)
              << ", " << bs_tmp_ptr->omega.at(1) << ", "
              << bs_tmp_ptr->omega.at(2) << "]" << std::endl;
    // Accelerometer
    std::cout << std::setprecision(4) << "acc: [" << bs_tmp_ptr->acc.at(0)
              << ", " << bs_tmp_ptr->acc.at(1) << ", " << bs_tmp_ptr->acc.at(2)
              << "]" << std::endl;
  }
  if (ms_tmp_ptr) {
    // Joint positions
    std::cout << "mot_pos: [";
    for (int i = 0; i < kNumMotors; ++i) {
      std::cout << std::setprecision(4) << ms_tmp_ptr->q.at(moti[i]) << ", ";
    }
    std::cout << "]" << std::endl;

    // Joint velocities
    std::cout << "mot_vel: [";
    for (int i = 0; i < kNumMotors; ++i) {
      std::cout << std::setprecision(4) << ms_tmp_ptr->dq.at(moti[i]) << ", ";
    }
    std::cout << "]" << std::endl;
  }
}

void HumanoidExample::UpdateTables(bool init) {
  // Clear the console
  std::cout << u8"\033[2J";

  if (init) {
    // Set tables border style
    table_IMU_.set_border_style(FT_NICE_STYLE);
    table_legs_.set_border_style(FT_NICE_STYLE);
    table_arms_.set_border_style(FT_NICE_STYLE);
    table_misc_.set_border_style(FT_NICE_STYLE);

    // Initialize headers
    table_IMU_.set_cur_cell(0, 0);
    table_legs_.set_cur_cell(0, 0);
    table_arms_.set_cur_cell(0, 0);
    table_misc_.set_cur_cell(0, 0);
    table_IMU_ << fort::header << ""
               << "X"
               << "Y"
               << "Z" << fort::endr;
    table_legs_ << fort::header << ""
                << "L Yaw"
                << "L Roll"
                << "L Pitch"
                << "L Knee"
                << "L Ank";
    table_legs_ << "R Yaw"
                << "R Roll"
                << "R Pitch"
                << "R Knee"
                << "R Ank" << fort::endr;
    table_arms_ << fort::header << ""
                << "L Pitch"
                << "L Roll"
                << "L Yaw"
                << "L Elbow";
    table_arms_ << "R Pitch"
                << "R Roll"
                << "R Yaw"
                << "R Elbow" << fort::endr;
    table_misc_ << fort::header << ""
                << "VX"
                << "VY"
                << "WZ" << fort::endr;
  }

  // Fill tables with data
  const std::shared_ptr<const BaseState> bs_tmp_ptr =
      base_state_buffer_.GetData();
  const std::shared_ptr<const MotorState> ms_tmp_ptr =
      motor_state_buffer_.GetData();

  // Set current cell to start of second row
  table_IMU_.set_cur_cell(1, 0);
  table_legs_.set_cur_cell(1, 0);
  table_arms_.set_cur_cell(1, 0);
  table_misc_.set_cur_cell(1, 0);

  // Fill IMU data
  if (bs_tmp_ptr) {
    table_IMU_ << "RPY";
    for (int i = 0; i < 3; ++i) {
      table_IMU_ << std::fixed << std::setprecision(4) << bs_tmp_ptr->rpy.at(i);
    }
    table_IMU_ << fort::endr << fort::separator << "Gyro";
    for (int i = 0; i < 3; ++i) {
      table_IMU_ << std::fixed << std::setprecision(4)
                 << bs_tmp_ptr->omega.at(i);
    }
    table_IMU_ << fort::endr << fort::separator << "Acc";
    for (int i = 0; i < 3; ++i) {
      table_IMU_ << std::fixed << std::setprecision(4) << bs_tmp_ptr->acc.at(i);
    }
  }

  // Fill joint data
  if (ms_tmp_ptr) {
    table_legs_ << "Pos";
    for (int i = 0; i < 10; ++i) {
      table_legs_ << std::fixed << std::setprecision(4)
                  << ms_tmp_ptr->q.at(moti[i]);
    }
    table_legs_ << fort::endr << fort::separator << "Vel";
    for (int i = 0; i < 10; ++i) {
      table_legs_ << std::fixed << std::setprecision(4)
                  << ms_tmp_ptr->dq.at(moti[i]);
    }
    table_legs_ << fort::endr << fort::separator << "Torques";
    for (int i = 0; i < 10; ++i) {
      table_legs_ << std::fixed << std::setprecision(4)
                  << ms_tmp_ptr->tau.at(moti[i]); // tau_des_[i];
    }
    table_legs_ << fort::endr;

    table_arms_ << "Pos";
    for (int i = 11; i < 19; ++i) {
      table_arms_ << std::fixed << std::setprecision(4)
                  << ms_tmp_ptr->q.at(moti[i]);
    }
    table_arms_ << fort::endr << fort::separator << "Vel";
    for (int i = 11; i < 19; ++i) {
      table_arms_ << std::fixed << std::setprecision(4)
                  << ms_tmp_ptr->dq.at(moti[i]);
    }
    table_arms_ << fort::endr << fort::separator << "Torques";
    for (int i = 11; i < 19; ++i) {
      table_arms_ << std::fixed << std::setprecision(4)
                  << ms_tmp_ptr->tau.at(moti[i]); // tau_des_[i];
    }
    table_arms_ << fort::endr;
  }

  table_misc_ << "Vel cmd" << std::fixed << std::setprecision(4) << cmd_(0)
              << cmd_(1) << cmd_(5) << fort::endr;

  if (init) {
    // Set text style
    table_IMU_.row(0).set_cell_content_text_style(fort::text_style::bold);
    table_IMU_.column(0).set_cell_content_text_style(fort::text_style::bold);
    table_legs_.column(0).set_cell_content_text_style(fort::text_style::bold);
    table_arms_.column(0).set_cell_content_text_style(fort::text_style::bold);
    table_misc_.row(0).set_cell_content_text_style(fort::text_style::bold);
    table_misc_.column(0).set_cell_content_text_style(fort::text_style::bold);

    // Set alignment
    table_IMU_.column(0).set_cell_text_align(fort::text_align::center);
    for (int i = 1; i < 4; ++i) {
      table_IMU_.column(i).set_cell_text_align(fort::text_align::right);
      table_IMU_.column(i).set_cell_min_width(9);
    }
    table_IMU_[0][1].set_cell_text_align(fort::text_align::center);
    table_IMU_[0][2].set_cell_text_align(fort::text_align::center);
    table_IMU_[0][3].set_cell_text_align(fort::text_align::center);

    table_legs_.column(0).set_cell_text_align(fort::text_align::center);
    for (int i = 1; i < 11; ++i) {
      table_legs_.column(i).set_cell_text_align(fort::text_align::right);
      table_legs_.column(i).set_cell_min_width(9);
    }

    table_arms_.column(0).set_cell_text_align(fort::text_align::center);
    for (int i = 1; i < 11; ++i) {
      table_arms_.column(i).set_cell_text_align(fort::text_align::right);
      table_arms_.column(i).set_cell_min_width(9);
    }

    table_misc_.column(0).set_cell_text_align(fort::text_align::center);
    for (int i = 1; i < 4; ++i) {
      table_misc_.column(i).set_cell_text_align(fort::text_align::right);
      table_misc_.column(i).set_cell_min_width(9);
    }
    table_misc_[0][1].set_cell_text_align(fort::text_align::center);
    table_misc_[0][2].set_cell_text_align(fort::text_align::center);
    table_misc_[0][3].set_cell_text_align(fort::text_align::center);
  }

  std::string buffer = "";
  std::string newLine = std::string("\n");
  switch (status_) {
  case STATUS_INIT:
    buffer += std::string("    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓") + newLine;
    buffer += std::string("    ┃      Initialization      ┃") + newLine;
    buffer += std::string("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛") + newLine + newLine;
    break;
  case STATUS_WAITING_AIR:
    buffer += std::string("    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓") + newLine;
    buffer += std::string("    ┃    Waiting in the air    ┃") + newLine;
    buffer += std::string("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛") + newLine + newLine;
    break;
  case STATUS_GAIN_TRANSITION:
    buffer += std::string("    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓") + newLine;
    buffer += std::string("    ┃   PD Gains Transition    ┃") + newLine;
    buffer += std::string("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛") + newLine + newLine;
    break;
  case STATUS_RUN:
    buffer += std::string("    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓") + newLine;
    buffer += std::string("    ┃    Running Controller    ┃") + newLine;
    buffer += std::string("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛") + newLine + newLine;
    break;
  case STATUS_DAMPING:
    buffer += std::string("    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓") + newLine;
    buffer += std::string("    ┃    Emergency Damping!    ┃") + newLine;
    buffer += std::string("    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛") + newLine + newLine;
    break;
  }
  buffer += std::string("    ┏━━━━━━━━━━━━━━━━━━━┓") + newLine;
  buffer += std::string("    ┃    Sensor Data    ┃") + newLine;
  buffer += std::string("    ┗━━━━━━━━━━━━━━━━━━━┛") + newLine + newLine;
  buffer += table_IMU_.to_string() + newLine;
  buffer += table_legs_.to_string() + newLine;
  buffer += table_arms_.to_string() + newLine;
  buffer += table_misc_.to_string() + newLine;
  buffer += std::string("Time: ") + std::to_string(time_) + newLine;
  std::cout << buffer << std::flush;
}

////
// UTILS
////

char *HumanoidExample::getCurrentDateTime() {
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S.txt");
  char *result = new char[ss.str().length() + 1];
  std::strcpy(result, ss.str().c_str());

  return result;
}