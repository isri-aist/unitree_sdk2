#pragma once

#include <array>
#include <stdint.h>

#include <unitree/idl/go2/LowCmd_.hpp>

constexpr int kNumMotors = 20;

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
  // Right leg
  kRightHipYaw = 8,
  kRightHipRoll = 0,
  kRightHipPitch = 1,
  kRightKnee = 2,
  kRightAnkle = 11,
  // Left leg
  kLeftHipYaw = 7,
  kLeftHipRoll = 3,
  kLeftHipPitch = 4,
  kLeftKnee = 5,
  kLeftAnkle = 10,

  kWaistYaw = 6,

  kNotUsedJoint = 9,

  // Right arm
  kRightShoulderPitch = 12,
  kRightShoulderRoll = 13,
  kRightShoulderYaw = 14,
  kRightElbow = 15,
  // Left arm
  kLeftShoulderPitch = 16,
  kLeftShoulderRoll = 17,
  kLeftShoulderYaw = 18,
  kLeftElbow = 19,

};

uint32_t Crc32Core(uint32_t *ptr, uint32_t len);
