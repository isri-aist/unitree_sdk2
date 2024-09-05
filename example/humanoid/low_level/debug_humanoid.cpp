#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string>
#include <thread>

#include "cpuMLP/Interface.hpp"
#include "cpuMLP/Types.h"

#include "Joystick.hpp"

int main() {

  using namespace std::chrono_literals;

  const Vector20 q_init{
      0.0, 0.0, -0.2, 0.6, -0.4, 0.0, 0.0, -0.2, 0.6,  -0.4, // Legs
      0.0, 0.4, 0.0,  0.0, -0.4, 0.4, 0.0, 0.0,  -0.4,       // Torso and arms
      0.0};                                                  // Unused joint

  // MLP interface
  Interface mlpInterface_(43, 0, 10, 1, 1);

  // Create link with MLP
  mlpInterface_.initialize(
      "/home/paleziart/git/policies/H1Terrain_08-01_11-22-18/nn/",
      q_init.head(10), 0.25);

  Vector10 pos = q_init.head(10);
  Vector10 vel = Vector10::Zero();
  Vector4 ori {0.0, 0.0, 0.0, 1.0};
  Vector3 gyro = Vector3::Zero();
  Vector6 cmd = Vector6::Zero();
  float time = 0.0;
  mlpInterface_.update_observation(pos.head(10), vel.head(10), ori, gyro, cmd, time);

  std::cout << "================" << std::endl;
  std::cout << mlpInterface_.forward() << std::endl;

  Joystick joy;
  joy.initialize(0.02);
  std::cout << joy.getVRef() << std::endl;

  while (true) {
    joy.update_v_ref();
    std::this_thread::sleep_for(20ms);
    std::cout << joy.getVRef().transpose() << std::endl;
  }

  return 0;
}