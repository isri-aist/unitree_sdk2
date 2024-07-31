#pragma once

#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string>

#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include <unitree/common/thread/thread.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>

#include "base_state.h"
#include "cpuMLP/Interface.hpp"
#include "cpuMLP/Types.h"
#include "data_buffer.hpp"
#include "lib/fort.c"
#include "lib/fort.hpp"
#include "motors.hpp"

#define FMT_HEADER_ONLY
#include "fmtlog-inl.h"
#include "fmtlog.h"

static const std::string kTopicLowCommand = "rt/lowcmd";
static const std::string kTopicLowState = "rt/lowstate";

template <typename T, size_t N>
std::string arrayToStringView(const std::array<T, N> &arr) {
  std::ostringstream oss;
  for (size_t i = 0; i < N; ++i) {
    oss << arr[i];
    if (i < N - 1) {
      oss << ",";
    }
  }
  return oss.str();
}

template <> struct fmt::formatter<BaseState> : formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(BaseState bs, FormatContext &ctx) {

    const std::string quat = "quat," + arrayToStringView(bs.quat);
    const std::string rpy = "rpy," + arrayToStringView(bs.rpy);
    const std::string omega = "omega," + arrayToStringView(bs.omega);
    const std::string acc = "acc," + arrayToStringView(bs.acc);
    const std::string all = quat + "\n" + rpy + "\n" + omega + "\n" + acc;

    string_view name = all;

    return formatter<string_view>::format(name, ctx);
  }
};

template <> struct fmt::formatter<MotorState> : formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(MotorState ms, FormatContext &ctx) {

    const std::string q = "q," + arrayToStringView(ms.q);
    const std::string dq = "dq," + arrayToStringView(ms.dq);
    const std::string all = q + "\n" + dq;

    string_view name = all;

    return formatter<string_view>::format(name, ctx);
  }
};

template <> struct fmt::formatter<MotorCommand> : formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(MotorCommand mc, FormatContext &ctx) {

    const std::string q_ref = "q_ref," + arrayToStringView(mc.q_ref);
    const std::string dq_ref = "dq_ref," + arrayToStringView(mc.dq_ref);
    const std::string kp = "kp," + arrayToStringView(mc.kp);
    const std::string kd = "kd," + arrayToStringView(mc.kd);
    const std::string tau_ff = "tau_ff," + arrayToStringView(mc.tau_ff);
    const std::string all =
        q_ref + "\n" + dq_ref + "\n" + kp + "\n" + kd + "\n" + tau_ff;

    string_view name = all;

    return formatter<string_view>::format(name, ctx);
  }
};

class HumanoidExample {
public:
  HumanoidExample(const std::string &networkInterface = "")
      : mlpInterface_(43, 0, 10, 1, 1) {
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
        &HumanoidExample::UpdateTables, this, false);

    // Define default configuration
    q_init_ << 0.0, 0.0, -0.2, 0.6, -0.4, 0.0, 0.0, -0.2, 0.6, -0.4, // Legs
        0.0, 0.4, 0.0, 0.0, -0.4, 0.4, 0.0, 0.0, -0.4, // Torso and arms
        0.0;                                           // Unused joint

    // Define Kp and Kd gains
    kp_.fill(kp_low_);
    kd_.fill(kd_low_);
    kp_.head(11) << 200, 200, 200, 300, 40, 200, 200, 200, 300, 40, kp_high_;
    kd_.head(11) << 5, 5, 5, 6, 2, 5, 5, 5, 6, 2, kd_high_;

    // Create link with MLP
    Vector7 scales {0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    mlpInterface_.initialize(
        "/home/paleziart/git/policies/policy-2024-07-01/nn/", q_init_.head(10),
        scales);

    // Initialize tables for console display
    UpdateTables(true);

    // Initialize sink for data logging
    fmtlog::setHeaderPattern("");
    fmtlog::setLogFile(getCurrentDateTime());
    fmtlog::setFlushDelay(100000000);
    fmtlog::startPollingThread(100000000);
  }

  ~HumanoidExample() = default;

  // Get the current date in local time
  char *getCurrentDateTime() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S.txt");
    char *result = new char[ss.str().length() + 1];
    std::strcpy(result, ss.str().c_str());

    return result;
  }

  // Prepare the command message and send it to the publisher
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

  // Update motor and base states using received sensor message
  void LowStateHandler(const void *message) {
    unitree_go::msg::dds_::LowState_ low_state =
        *(unitree_go::msg::dds_::LowState_ *)message;

    RecordMotorState(low_state);
    RecordBaseState(low_state);
  }

  // Take decisions for the next commands and send them to the motor command buffer
  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms_tmp_ptr =
        motor_state_buffer_.GetData();

    if (ms_tmp_ptr) {
      time_ += control_dt_;

      if (false) { // (time_ > init_duration_) {
        const std::shared_ptr<const BaseState> bs_tmp_ptr =
            base_state_buffer_.GetData();
        const std::shared_ptr<const MotorState> ms_tmp_ptr =
            motor_state_buffer_.GetData();

        Vector10 pos, vel;
        for (int i = 0; i < 10; ++i) {
          pos(i) = ms_tmp_ptr->q.at(moti[i]);
          vel(i) = ms_tmp_ptr->dq.at(moti[i]);
        }
        Vector4 ori(bs_tmp_ptr->quat.data());
        Vector3 gyro(bs_tmp_ptr->omega.data());
        mlpInterface_.update_observation(pos, vel, ori, gyro, time_);

        std::cout
            << std::setprecision(4)
            << mlpInterface_.historyObs_.head(mlpInterface_.obsDim_).transpose()
            << std::endl;

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
          if (i == JointIndex::kLeftHipPitch ||
              i == JointIndex::kRightHipPitch) {
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

      // Log sensors and commands
      LogAll();
    }
  }

  // Print sensor data to the terminal
  void ReportSensors() {
    const std::shared_ptr<const BaseState> bs_tmp_ptr =
        base_state_buffer_.GetData();
    const std::shared_ptr<const MotorState> ms_tmp_ptr =
        motor_state_buffer_.GetData();
    if (bs_tmp_ptr) {
      // Roll Pitch Yaw orientation
      std::cout << std::setprecision(4) << "rpy: [" << bs_tmp_ptr->rpy.at(0)
                << ", " << bs_tmp_ptr->rpy.at(1) << ", "
                << bs_tmp_ptr->rpy.at(2) << "]" << std::endl;
      // Gyroscope
      std::cout << std::setprecision(4) << "gyro: [" << bs_tmp_ptr->omega.at(0)
                << ", " << bs_tmp_ptr->omega.at(1) << ", "
                << bs_tmp_ptr->omega.at(2) << "]" << std::endl;
      // Accelerometer
      std::cout << std::setprecision(4) << "acc: [" << bs_tmp_ptr->acc.at(0)
                << ", " << bs_tmp_ptr->acc.at(1) << ", "
                << bs_tmp_ptr->acc.at(2) << "]" << std::endl;
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

  void UpdateTables(bool init=false) {

    // Clear the console
    std::cout << u8"\033[2J";

    if(init) {
      // Set tables border style
      table_IMU_.set_border_style(FT_NICE_STYLE);
      table_joints_.set_border_style(FT_NICE_STYLE);

      // Initialize headers
      table_IMU_.set_cur_cell(0, 0);
      table_joints_.set_cur_cell(0, 0);
      table_IMU_ << fort::header << "" << "X" << "Y" << "Z" << fort::endr;
      table_joints_ << fort::header << "" << "L Yaw" << "L Roll" << "L Pitch" << "L Knee" << "L Ank";
      table_joints_ << "R Yaw" << "R Roll" << "R Pitch" << "R Knee" << "R Ank" << fort::endr;
    }

    // Fill tables with data
    const std::shared_ptr<const BaseState> bs_tmp_ptr =
        base_state_buffer_.GetData();
    const std::shared_ptr<const MotorState> ms_tmp_ptr =
        motor_state_buffer_.GetData();

    // Set current cell to start of second row
    table_IMU_.set_cur_cell(1, 0);
    table_joints_.set_cur_cell(1, 0);

    // Fill IMU data
    if (bs_tmp_ptr) {
      table_IMU_ << "RPY";
      for (int i = 0; i < 3; ++i) {
        table_IMU_ << std::fixed << std::setprecision(4) << bs_tmp_ptr->rpy.at(i);
      }
      table_IMU_ << fort::endr << fort::separator << "Gyro";
      for (int i = 0; i < 3; ++i) {
        table_IMU_ << std::fixed << std::setprecision(4) << bs_tmp_ptr->omega.at(i);
      }
      table_IMU_ << fort::endr << fort::separator << "Acc";
      for (int i = 0; i < 3; ++i) {
        table_IMU_ << std::fixed << std::setprecision(4) << bs_tmp_ptr->acc.at(i);
      }
    }

    // Fill joint data
    if (ms_tmp_ptr) {
      table_joints_ << "Pos";
      for (int i = 0; i < 10; ++i) {
        table_joints_ << std::fixed << std::setprecision(4) << ms_tmp_ptr->q.at(moti[i]);
      }
      table_joints_ << fort::endr << fort::separator << "Vel";
      for (int i = 0; i < 10; ++i) {
        table_joints_ << std::fixed << std::setprecision(4) << ms_tmp_ptr->dq.at(moti[i]);
      }
      table_joints_ << fort::endr;
    }

    if (init) {
      // Set text style
      table_IMU_.row(0).set_cell_content_text_style(fort::text_style::bold);
      table_IMU_.column(0).set_cell_content_text_style(fort::text_style::bold);
      table_joints_.column(0).set_cell_content_text_style(fort::text_style::bold);

      // Set alignment
      table_IMU_.column(0).set_cell_text_align(fort::text_align::center);
      for (int i = 1; i < 4; ++i) {
          table_IMU_.column(i).set_cell_text_align(fort::text_align::right);
          table_IMU_.column(i).set_cell_min_width(9);
      }
      table_IMU_[0][1].set_cell_text_align(fort::text_align::center);
      table_IMU_[0][2].set_cell_text_align(fort::text_align::center);
      table_IMU_[0][3].set_cell_text_align(fort::text_align::center);

      table_joints_.column(0).set_cell_text_align(fort::text_align::center);
      for (int i = 1; i < 11; ++i) {
          table_joints_.column(i).set_cell_text_align(fort::text_align::right);
          table_joints_.column(i).set_cell_min_width(9);
      }
    }

    std::cout << "    ┏━━━━━━━━━━━━━━━━━━━┓" << std::endl;
    std::cout << "    ┃    Sensor Data    ┃" << std::endl;
    std::cout << "    ┗━━━━━━━━━━━━━━━━━━━┛" << std::endl << std::endl;
    std::cout << table_IMU_.to_string() << std::endl;
    std::cout << table_joints_.to_string() << std::endl;
  }

  void LogAll() {

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

  // Table for console display
  fort::char_table table_IMU_;
  fort::char_table table_joints_;

};
