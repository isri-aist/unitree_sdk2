#pragma once

#include <array>
#include <stdint.h>

#include <unitree/idl/go2/LowCmd_.hpp>

constexpr int kNumMotors = 12;

struct MotorCommand {
  std::array<float, kNumMotors> q_ref = {};
  std::array<float, kNumMotors> dq_ref = {};
  std::array<float, kNumMotors> kp = {};
  std::array<float, kNumMotors> kd = {};
  std::array<float, kNumMotors> tau_ff = {};
};

struct MotorState {
  std::array<float, kNumMotors> q = {};
  std::array<float, kNumMotors> dq = {};
  std::array<float, kNumMotors> tau = {};
};

enum JointIndex {
  FL_hip_joint = 0,
  FL_thigh_joint = 1,
  FL_calf_joint = 2,
  FR_hip_joint = 3,
  FR_thigh_joint = 4,
  FR_calf_joint = 5,
  RL_hip_joint = 6,
  RL_thigh_joint = 7,
  RL_calf_joint = 8,
  RR_hip_joint = 9,
  RR_thigh_joint = 10,
  RR_calf_joint = 11
};

uint32_t Crc32Core(uint32_t *ptr, uint32_t len);
