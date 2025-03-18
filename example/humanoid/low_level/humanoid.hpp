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

#include "Interface.hpp"
#include "Joystick.hpp"
#include "Types.h"
#include "base_state.h"
#include "data_buffer.hpp"
#include "lib/fort.c"
#include "lib/fort.hpp"
#include "logger.hpp"
#include "motors.hpp"

#define USE_JOYSTICK true

#define STATUS_INIT 0
#define STATUS_WAITING_AIR 1
#define STATUS_WAITING_GRD 2
#define STATUS_GAIN_TRANSITION 3
#define STATUS_RUN 4
#define STATUS_DAMPING 5

static const std::string kTopicLowCommand = "rt/lowcmd";
static const std::string kTopicLowState = "rt/lowstate";

class HumanoidExample;
void waiting(HumanoidExample *HE);

class HumanoidExample {
public:
  HumanoidExample(const std::string &networkInterface = "",
                  const std::string &model_file = "")
      : mlpInterface_() {
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);
    std::cout << "Initialize channel factory." << std::endl;

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

    // Create link with network interface
    mlpInterface_.initialize(model_file, q_init_.head(19), control_dt_);
    policy_out_ = Vxf::Zero(mlpInterface_.get_actDim());

    // Initialize tables for console display
    UpdateTables(true);

    // Initialize sink for data logging
    fmtlog::setHeaderPattern("");
    fmtlog::setLogFile(getCurrentDateTime());
    fmtlog::setFlushDelay(100000000);
    fmtlog::startPollingThread(100000000);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Default destructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~HumanoidExample() = default;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Get the current date in local time
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  char *getCurrentDateTime();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Transform quat orientation into projected gravity vector
  ///
  /// \param[in] _bodyQuat The orientation expressed as a (x, y, z, w)
  /// quaternion
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void transformBodyQuat(Vector4 _bodyQuat);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Prepare the command message and send it to the publisher
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void LowCommandWriter();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update motor and base states using received sensor message
  ///
  /// \param[in] message State message containing information from the sensors
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void LowStateHandler(const void *message);

  // Take decisions for the next commands and send them to the motor command
  // buffer
  void Control() {
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
      const bool lim_lower = ((pos - 0.85 * q_lim_lower).array() < 0.0).any();
      const bool lim_upper = ((pos - 0.85 * q_lim_upper).array() > 0.0).any();
      if (lim_lower || lim_upper) {
        std::cout << "Joint range threshold breached!!!" << std::endl;
        status_ = STATUS_DAMPING;
      }

      // Check if joint velocities are too high
      const bool lim_velocity = ((vel.array().abs() - qdot_limit.array()) > 0.0).any();
      if (lim_velocity) {
        std::cout << "Velocity threshold breached!!!" << std::endl;
        std::cout << "VEL: " << std::fixed << std::setprecision(4) << vel.transpose() << std::endl;
        status_ = STATUS_DAMPING;
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
        policy_out_ = mlpInterface_.forward();
        for (int i = 0; i < 10; ++i) {
          policy_log_[i] = policy_out_[i];
        }

        if (time_run_ < interp_duration_) {
          break;
        }
        */

        // Refresh joystick
        if (USE_JOYSTICK) {
          joy_.update_v_ref();
          cmd_ = joy_.getVRef();
        } else {
          cmd_ = Vector6::Zero();
        }

        cmd_ << 0, 0, 1, 0, 0, 0;
        if (USE_JOYSTICK) {
          if (joy_.getCross()) {
            float yaw_vref = joy_.getVRef()[1];
            cmd_ << 0, 1, 0, yaw_vref, 0, 0;
          }
          if (joy_.getSquare()) {
            float vx_vref = joy_.getVRef()[0];
            float vy_vref = joy_.getVRef()[1];
            cmd_ << 1, 0, 0, 0, vx_vref, vy_vref;
          }
          if (joy_.getCircle()) {
            cmd_ << Vector6::Zero();
            status_ = STATUS_DAMPING;
          }
        }

        Vector3 rpy(bs_tmp_ptr->rpy.data());
        Vector4 ori(bs_tmp_ptr->quat.data());
        Vector3 gyro(bs_tmp_ptr->omega.data());

        /*
        // Update observation vector (ManiSkill)
        mlpInterface_.update_observation_ManiSkill(pos.head(19), vel.head(19), tau.head(19), rpy,
                                                   quatPermut * ori, gyro, cmd_, time_run_);

        // Inference to get position targets from the policy (ManiSkill)
        policy_out_ = mlpInterface_.forward_ManiSkill();
        */

        // Update observation vector (Mujoco)
        // mlpInterface_.update_observation_with_clock(pos.head(19), vel.head(19), tau.head(19), rpy,
        //                                             quatPermut * ori,  gyro, cmd_, 0.5 + time_run_);

        mlpInterface_.update_full_body_observation(pos.head(19), vel.head(19), tau.head(19), rpy, gyro, time_run_);
        
        // Inference to get position targets from the policy (Mujoco)
        policy_out_ = mlpInterface_.forward();

        // Check policy output size
        //assert(policy_out_.rows() == 10);
	assert(policy_out_.rows() == mlpInterface_.get_actDim());

        // Logging policy output
        for (int i = 0; i < policy_out_.rows(); ++i) {
          policy_log_[i] = policy_out_[i];
        }

        // Send policy commands to the robot
        Vxf network_cmd = policy_out_;
        float q_des = 0.f;
        for (int i = 0; i < kNumMotors; ++i) {
          q_des = i < mlpInterface_.get_actDim() ? network_cmd(i) : q_init_(i);
          motor_command_tmp.kp.at(moti[i]) = kp_(i);
          motor_command_tmp.kd.at(moti[i]) = kd_(i);
          motor_command_tmp.q_ref.at(moti[i]) = q_des;
          motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
          motor_command_tmp.tau_ff.at(moti[i]) = 0.f;
        }
        break;
      }
      case STATUS_WAITING_AIR: {
        // Wait at default configuration
        for (int i = 0; i < kNumMotors; ++i) {
          motor_command_tmp.kp.at(moti[i]) = kp_wait_(i);
          motor_command_tmp.kd.at(moti[i]) = kd_wait_(i);
          motor_command_tmp.q_ref.at(moti[i]) = new_q_init_(i);
          motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
          motor_command_tmp.tau_ff.at(moti[i]) = 0.f;
        }
        break;
      }
      case STATUS_WAITING_GRD: {
        // Wait at default configuration
        for (int i = 0; i < kNumMotors; ++i) {
          motor_command_tmp.kp.at(moti[i]) = kp_wait_(i);
          motor_command_tmp.kd.at(moti[i]) = kd_wait_(i);
          motor_command_tmp.q_ref.at(moti[i]) = new_q_init_(i);
          motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
          motor_command_tmp.tau_ff.at(moti[i]) = tau_ff_(i) * 0.0;
        }
        break;
      }
      case STATUS_GAIN_TRANSITION: {

        bool start = true;
        if (USE_JOYSTICK) {
          joy_.update_v_ref();
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
          motor_command_tmp.q_ref.at(moti[i]) = new_q_init_(i);
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

          float q_des = (new_q_init_(i) - ms_tmp_ptr->q.at(moti[i])) * ratio +
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
        tau_des_[i] =
            motor_command_tmp.kp.at(i) *
                (motor_command_tmp.q_ref.at(i) - ms_tmp_ptr->q.at(i)) +
            motor_command_tmp.kd.at(i) *
                (motor_command_tmp.dq_ref.at(i) - ms_tmp_ptr->dq.at(i)) +
            motor_command_tmp.tau_ff.at(i);
      }
      LogAll();
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Basic print of sensor data to the console
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void ReportSensors();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Launch controller once Enter is pressed
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void endWaiting();

private:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Refresh the quantities in the tables displayed in the console
  ///
  /// \param[in] init Initialize style and header of tables
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void UpdateTables(bool init = false);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Log all monitored quantities for the current time step
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void LogAll();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Read and record information about low level motor states (pos, vel,
  /// torques)
  ///
  /// \param[in] msg Low-state message containing information about the motors
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void RecordMotorState(const unitree_go::msg::dds_::LowState_ &msg);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Read and record information about low level motor states (pos, vel,
  /// torques)
  ///
  /// \param[in] msg Low-state message containing information about the motors
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void RecordBaseState(const unitree_go::msg::dds_::LowState_ &msg);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Check if a motor index corresponds to a "weak" motor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
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

  unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>
      lowcmd_publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>
      lowstate_subscriber_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<BaseState> base_state_buffer_;

  // MLP interface
  Interface mlpInterface_;

  // control params
  const float control_dt_ = 0.025f;

  int status_ = STATUS_INIT;

  // Default configuration
  const Vector20 new_q_init_{
      // -0.2, -0.05, -0.18, 0.74, -0.51, -0.06, 0.06, -0.48,  1.1, -0.27,
      0.0, 0.0, -0.2, 0.6, -0.4, 0.0, 0.0, -0.2,  0.6, -0.4,
      0.01,
      // -0.23, 0.18, -1.07, 1.44, -0.13, -0.12, 0.93, 1.42,
      0.4,  0.0, 0.0, -0.4, 0.4, 0.0,  0.0, -0.4,       // Torso and arms
      0};

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
      0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, // Legs
      2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3,  2.61, // Torso and arms
      0.0};                                                 // Unused joint
  const Vector20 qdot_limit{
      8, 8, 8, 8, 8,
      8, 8, 8, 8, 8,  // Legs
      4,  // Torso
      12, 12, 12, 12,
      12, 12, 12, 12, // Arms
      0 // Unsused joint
  };

  // Proportional derivative gains
  Vector20 kp_{100.0, 100.0, 100.0, 100.0, 20.0,
               100.0, 100.0, 100.0, 100.0, 20.0, // Legs
               40.0, // Torso
	       20.0, 20.0, 20.0, 20.0,
               20.0, 20.0, 20.0, 20.0, // Arms
               0.0};                       // Unused joint

  Vector20 kd_{10.0, 10.0, 10.0, 10.0, 4.0, 10.0, 10.0, 10.0, 10.0, 4.0, // Legs
               4.0, // Torso
	       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // Arms
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

  Vector20 tau_ff_{0.0, 6.0, -8.0, -26.0, 36.0, 0.0, -6.0, -8.0, -26.0, 36.0,
                   0.0, 0.0, 0.0,  0.0,   0.0,  0.0, 0.0,  0.0,  0.0,   0.0};

  std::array<float, kNumMotors> tau_des_ = {};
  std::array<float, 19> policy_log_ = {};

  Vector6 cmd_ = Vector6::Zero();

  Vxf policy_out_;

  const Eigen::Matrix<float, 4, 4> quatPermut{{0, 1, 0, 0},
                                              {0, 0, 1, 0},
                                              {0, 0, 0, 1},
                                              {1, 0, 0, 0}}; // Reorder quat vector

  float time_ = 0.f;
  float time_run_ = 0.f;
  const float init_duration_ = 5.f;
  const float interp_duration_ = .1f;

  float report_dt_ = 0.1f;

  // Joystick interface
  Joystick joy_;

  // multithreading
  unitree::common::ThreadPtr command_writer_ptr_;
  unitree::common::ThreadPtr control_thread_ptr_;
  unitree::common::ThreadPtr report_sensors_ptr_;

  // Table for console display
  fort::char_table table_IMU_;
  fort::char_table table_legs_;
  fort::char_table table_arms_;
  fort::char_table table_misc_;
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
    status_ = STATUS_WAITING_GRD;
    std::thread wait_thread(waiting, this);
    wait_thread.detach();
  } else if (status_ == STATUS_WAITING_GRD) {
    time_run_ = -control_dt_;
    status_ = STATUS_GAIN_TRANSITION;
  }
}

////
// READ / WRITE MESSAGES
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

void HumanoidExample::RecordMotorState(
    const unitree_go::msg::dds_::LowState_ &msg) {
  MotorState ms_tmp;
  for (int i = 0; i < kNumMotors; ++i) {
    ms_tmp.q.at(i) = msg.motor_state()[i].q();
    ms_tmp.dq.at(i) = msg.motor_state()[i].dq();
    ms_tmp.tau.at(i) = msg.motor_state()[i].tau_est();
  }

  motor_state_buffer_.SetData(ms_tmp);
}

void HumanoidExample::RecordBaseState(
    const unitree_go::msg::dds_::LowState_ &msg) {
  BaseState bs_tmp;
  bs_tmp.omega = msg.imu_state().gyroscope();
  bs_tmp.quat = msg.imu_state().quaternion();
  bs_tmp.rpy = msg.imu_state().rpy();
  bs_tmp.acc = msg.imu_state().accelerometer();

  base_state_buffer_.SetData(bs_tmp);
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

  // Log all monitored variables
  logi("time,{}", time_);
  if (ms_tmp_ptr) {
    logi("{}", *ms_tmp_ptr);
  }
  if (mc_tmp_ptr) {
    logi("{}", *mc_tmp_ptr);
  }
  if (bs_tmp_ptr) {
    logi("{}", *bs_tmp_ptr);
  }
  logi("{}", "tau_des," + arrayToStringView(tau_des_));
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

  switch (status_) {
  case STATUS_INIT:
    std::cout << "    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓" << std::endl;
    std::cout << "    ┃      Initialization      ┃" << std::endl;
    std::cout << "    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛" << std::endl << std::endl;
    break;
  case STATUS_WAITING_AIR:
    std::cout << "    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓" << std::endl;
    std::cout << "    ┃    Waiting in the air    ┃" << std::endl;
    std::cout << "    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛" << std::endl << std::endl;
    break;
  case STATUS_WAITING_GRD:
    std::cout << "    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓" << std::endl;
    std::cout << "    ┃   Waiting on the ground  ┃" << std::endl;
    std::cout << "    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛" << std::endl << std::endl;
    break;
  case STATUS_GAIN_TRANSITION:
    std::cout << "    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓" << std::endl;
    std::cout << "    ┃   PD Gains Transition    ┃" << std::endl;
    std::cout << "    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛" << std::endl << std::endl;
    break;
  case STATUS_RUN:
    std::cout << "    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓" << std::endl;
    std::cout << "    ┃    Running Controller    ┃" << std::endl;
    std::cout << "    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛" << std::endl << std::endl;
    break;
  case STATUS_DAMPING:
    std::cout << "    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓" << std::endl;
    std::cout << "    ┃    Emergency Damping!    ┃" << std::endl;
    std::cout << "    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛" << std::endl << std::endl;
    break;
  }
  std::cout << "    ┏━━━━━━━━━━━━━━━━━━━┓" << std::endl;
  std::cout << "    ┃    Sensor Data    ┃" << std::endl;
  std::cout << "    ┗━━━━━━━━━━━━━━━━━━━┛" << std::endl << std::endl;
  std::cout << table_IMU_.to_string() << std::endl;
  std::cout << table_legs_.to_string() << std::endl;
  std::cout << table_arms_.to_string() << std::endl;
  std::cout << table_misc_.to_string() << std::endl;
  std::cout << "Time: " << time_ << std::endl;
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

void HumanoidExample::transformBodyQuat(Vector4 _bodyQuat) {
  Vector3 _gravityVec, _qa, _qb, _qc, _qvec, _bodyOri;
  float q_w = 0.0;
  _gravityVec << 0.0, 0.0, -1.;

  // Body QUAT and gravity vector of 0 , 0, -1
  q_w = _bodyQuat[3];
  _qvec = _bodyQuat.head(3);
  _qa = _gravityVec * (2. * q_w * q_w - 1.);
  _qb = _qvec.cross(_gravityVec) * q_w * 2.0;
  _qc = _qvec * (_qvec.transpose() * _gravityVec) * 2.0;
  _bodyOri = _qa - _qb + _qc;
}
