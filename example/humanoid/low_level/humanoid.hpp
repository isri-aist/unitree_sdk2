#pragma once

#include <iostream>
#include <stdint.h>
#include <string>

#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include <unitree/common/thread/thread.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>

#include "base_state.h"
#include "data_buffer.hpp"
#include "motors.hpp"
#include "cpuMLP/Types.h"
#include "cpuMLP/Interface.hpp"

static const std::string kTopicLowCommand = "rt/lowcmd";
static const std::string kTopicLowState = "rt/lowstate";

class HumanoidExample {
public:
  HumanoidExample(const std::string &networkInterface = "") : mlpInterface_(43, 0, 10, 1, 1) {
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);
    std::cout << "Initialize channel factory." << std::endl;

    /*lowcmd_publisher_.reset(
        new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(
            kTopicLowCommand));
    lowcmd_publisher_->InitChannel();
    command_writer_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "command_writer", UT_CPU_ID_NONE, 2000,
        &HumanoidExample::LowCommandWriter, this);*/

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
        &HumanoidExample::ReportSensors, this);

    // Define default configuration
    q_init_ << 0.0, 0.0, -0.2, 0.6, -0.4, 0.0, 0.0, -0.2, 0.6, -0.4, // Legs
               0.0, 0.4, 0.0, 0.0, -0.4, 0.4, 0.0, 0.0, -0.4,  // Torso and arms
               0.0;  // Unused joint

    // Define Kp and Kd gains
    kp_.fill(kp_low_);
    kd_.fill(kd_low_);
    kp_.head(11) << 200, 200, 200, 300, 40, 200, 200, 200, 300, 40, kp_high_;
    kd_.head(11) << 5, 5, 5, 6, 2, 5, 5, 5, 6, 2, kd_high_;

    // Create link with MLP
    Vector7 scales;
    scales << 0.5, 1.0, 1.0, 0.05, 2.0, 0.25, 5.0;
    mlpInterface_.initialize(
      "/home/paleziart/git/policies/policy-2024-07-01/nn/", q_init_.head(10), scales
    );

  }

  ~HumanoidExample() = default;

  void LowCommandWriter() {
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

  void LowStateHandler(const void *message) {
    unitree_go::msg::dds_::LowState_ low_state =
        *(unitree_go::msg::dds_::LowState_ *)message;

    RecordMotorState(low_state);
    RecordBaseState(low_state);
  }

  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms_tmp_ptr =
        motor_state_buffer_.GetData();

    if (ms_tmp_ptr) {
      time_ += control_dt_;

      if (time_ > init_duration_) {
        const std::shared_ptr<const BaseState> bs_tmp_ptr = base_state_buffer_.GetData();
        const std::shared_ptr<const MotorState> ms_tmp_ptr = motor_state_buffer_.GetData();

        Vector10 pos, vel;
        for (int i = 0; i < 10; ++i) {
          pos(i) = ms_tmp_ptr->q.at(i);
          vel(i) = ms_tmp_ptr->dq.at(i);
        }
        Vector4 ori(bs_tmp_ptr->quat.data());
        Vector3 gyro(bs_tmp_ptr->omega.data());
        mlpInterface_.update_observation(pos, vel, ori, gyro, time_);

        std::cout << std::setprecision(4) << mlpInterface_.historyObs_.head(mlpInterface_.obsDim_).transpose() << std::endl;

        Vector10 network_cmd = q_init_.head(10);
        float q_des = 0.f;
        for (int i = 0; i < kNumMotors; ++i) {
          q_des = i < 10 ? network_cmd(i) : q_init_(i);
          motor_command_tmp.kp.at(moti[i]) = kp_(i);
          motor_command_tmp.kd.at(moti[i]) = kd_(i);
          motor_command_tmp.q_ref.at(moti[i]) = q_des;
          motor_command_tmp.dq_ref.at(moti[i]) = 0.f;
          motor_command_tmp.tau_ff.at(moti[i]) = 0.f;
        }

      } else {
        // Slowly move to default configuration

        float ratio = std::clamp(time_, 0.f, init_duration_) / init_duration_;
        for (int i = 0; i < kNumMotors; ++i) {
          motor_command_tmp.kp.at(i) = IsWeakMotor(i) ? kp_low_ : kp_high_;
          motor_command_tmp.kd.at(i) = IsWeakMotor(i) ? kd_low_ : kd_high_;
          motor_command_tmp.dq_ref.at(i) = 0.f;
          motor_command_tmp.tau_ff.at(i) = 0.f;

          float q_des = 0.f;
          if (i == JointIndex::kLeftHipPitch || i == JointIndex::kRightHipPitch) {
            q_des = hip_pitch_init_pos_;
          }
          if (i == JointIndex::kLeftKnee || i == JointIndex::kRightKnee) {
            q_des = knee_init_pos_;
          }
          if (i == JointIndex::kLeftAnkle || i == JointIndex::kRightAnkle) {
            q_des = ankle_init_pos_;
          }
          if (i == JointIndex::kLeftShoulderPitch ||
              i == JointIndex::kRightShoulderPitch) {
            q_des = shoulder_pitch_init_pos_;
          }

          q_des = (q_des - ms_tmp_ptr->q.at(i)) * ratio + ms_tmp_ptr->q.at(i);
          motor_command_tmp.q_ref.at(i) = q_des;
        }
      }
      // Write to command buffer
      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }

  void ReportSensors() {
    const std::shared_ptr<const BaseState> bs_tmp_ptr =
        base_state_buffer_.GetData();
    const std::shared_ptr<const MotorState> ms_tmp_ptr =
        motor_state_buffer_.GetData();
    if (bs_tmp_ptr) {
      // Roll Pitch Yaw orientation
      std::cout << std::setprecision(4) << "rpy: [" << bs_tmp_ptr->rpy.at(0) << ", "
                << bs_tmp_ptr->rpy.at(1) << ", " << bs_tmp_ptr->rpy.at(2) << "]"
                << std::endl;
      // Gyroscope
      std::cout << std::setprecision(4) << "gyro: [" << bs_tmp_ptr->omega.at(0) << ", "
                << bs_tmp_ptr->omega.at(1) << ", " << bs_tmp_ptr->omega.at(2) << "]"
                << std::endl;
      // Accelerometer
      std::cout << std::setprecision(4) << "acc: [" << bs_tmp_ptr->acc.at(0) << ", "
                << bs_tmp_ptr->acc.at(1) << ", " << bs_tmp_ptr->acc.at(2) << "]"
                << std::endl;
    }
    if (ms_tmp_ptr) {
      // Joint positions
      std::cout << "mot_pos: [";
      for (int i = 0; i < kNumMotors; ++i) {
        std::cout << std::setprecision(4) << ms_tmp_ptr->q.at(i) << ", ";
      }
      std::cout << "]" << std::endl;

      // Joint velocities
      std::cout << "mot_vel: [";
      for (int i = 0; i < kNumMotors; ++i) {
        std::cout << std::setprecision(4) << ms_tmp_ptr->dq.at(i) << ", ";
      }
      std::cout << "]" << std::endl;
    }
  }

private:
  void RecordMotorState(const unitree_go::msg::dds_::LowState_ &msg) {
    MotorState ms_tmp;
    for (int i = 0; i < kNumMotors; ++i) {
      ms_tmp.q.at(i) = msg.motor_state()[i].q();
      ms_tmp.dq.at(i) = msg.motor_state()[i].dq();
    }

    motor_state_buffer_.SetData(ms_tmp);
  }

  void RecordBaseState(const unitree_go::msg::dds_::LowState_ &msg) {
    BaseState bs_tmp;
    bs_tmp.omega = msg.imu_state().gyroscope();
    bs_tmp.quat = msg.imu_state().quaternion();
    bs_tmp.rpy = msg.imu_state().rpy();
    bs_tmp.acc = msg.imu_state().accelerometer();

    base_state_buffer_.SetData(bs_tmp);
  }

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
  float kp_low_ = 60.f;
  float kp_high_ = 200.f;
  float kd_low_ = 1.5f;
  float kd_high_ = 5.f;
  Vector20 kd_, kp_;

  float control_dt_ = 0.02f;

  float hip_pitch_init_pos_ = -0.5f;
  float knee_init_pos_ = 1.f;
  float ankle_init_pos_ = -0.5f;
  float shoulder_pitch_init_pos_ = 0.4f;
  Vector20 q_init_;

  float time_ = 0.f;
  float init_duration_ = 10.f;

  float report_dt_ = 0.1f;

  // multithreading
  unitree::common::ThreadPtr command_writer_ptr_;
  unitree::common::ThreadPtr control_thread_ptr_;
  unitree::common::ThreadPtr report_sensors_ptr_;
};
