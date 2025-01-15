#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string>
#include <thread>
#include <chrono>

#include "Types.h"
#include "Interface.hpp"
#include "Joystick.hpp"

int main(int argc, ORTCHAR_T* argv[]) {

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::microseconds;

  std::cout << "Hello there!" << std::endl;
  
  if (argc != 2) {
    std::cout << "Usage: ./onnx-api-example <onnx_model.onnx>" << std::endl;
    return -1;
  }

  std::basic_string<ORTCHAR_T> model_file = argv[1];
  
  OnnxWrapper wrapper(model_file);
  wrapper.initialize();

  // Run one inference
  VectorM A = wrapper.run(Eigen::VectorXf::Zero(28 * 28));
  std::cout << "Output:" << std::endl << A.transpose() << std::endl;

  // Time the average computation time for inference
  auto t1 = high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    wrapper.run(Eigen::VectorXf::Zero(28 * 28));
  }
  auto t2 = high_resolution_clock::now();

  // Getting number of microseconds as a double.
  duration<double, std::micro> ms_double = t2 - t1;

  std::cout << "Average computation time: " << ms_double.count() / 100 << " us\n";

  return 0;

  using namespace std::chrono_literals;

  const Vector20 q_init{
      0.0, 0.0, -0.2, 0.6, -0.4, 0.0, 0.0, -0.2, 0.6,  -0.4, // Legs
      0.0, 0.4, 0.0,  0.0, -0.4, 0.4, 0.0, 0.0,  -0.4,       // Torso and arms
      0.0};                                                  // Unused joint

  // MLP interface
  Interface mlpInterface_(51, 0, 10, 1, 1);

  Vector10 q_init_cut = q_init.head(10);
  // Vector4 q_arms; q_arms << 0.4, -0.4, 0.4, -0.4;
  // q_init_cut.block(10, 0, 4, 1) << 0.4, -0.4, 0.4, -0.4;

  // Create link with MLP
  mlpInterface_.initialize(
      "/home/paleziart/git/policies/H1Terrain_10-01_18-04-38/nn/",
      q_init_cut, 0.5, 0.02);

  Vector10 pos = q_init.head(10);
  pos << 0.0, 0.0, -0.2, 0.6, -0.4, 0.0, 0.0, -0.2, 0.6,  -0.4; // , 0.4, 0.0, 0.4, 0.0;
  Vector10 vel = Vector10::Zero();
  Vector10 tau = Vector10::Zero();
  Vector4 ori{0.0, 0.0, 0.0, 1.0};
  Vector3 rpy = Vector3::Zero();
  Vector3 gyro = Vector3::Zero();
  Vector6 cmd = Vector6::Zero();
  float time = 0.0;

  std::cout << "Q" << std::endl << pos.transpose() << std::endl;
  std::cout << "dQ" << std::endl << vel.transpose() << std::endl;
  std::cout << "TAU" << std::endl << tau.transpose() << std::endl;
  std::cout << "ORI" << std::endl << ori.transpose() << std::endl;
  std::cout << "GYRO" << std::endl << gyro.transpose() << std::endl;
  std::cout << "CMD" << std::endl << cmd.transpose() << std::endl; 

  for (int i = 0; i < 2; i++) {

    std::cout << "---- " << i << std::endl;

    mlpInterface_.update_observation(pos.head(10), vel.head(10), tau.head(10), rpy, gyro,
                                     cmd, time);

    mlpInterface_.forward();

  }
  return 0;

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