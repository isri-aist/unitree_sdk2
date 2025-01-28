#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string>
#include <thread>

#include "Interface.hpp"
#include "Joystick.hpp"
#include "Types.h"

int main(int argc, ORTCHAR_T *argv[]) {

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::microseconds;

  std::cout << "Hello there!" << std::endl;

  if (argc != 2) {
    std::cout << "Usage: ./bin/h1_test_example <onnx_model.onnx>" << std::endl;
    return -1;
  }

  std::basic_string<ORTCHAR_T> model_file = argv[1];

  std::unique_ptr<OnnxWrapper> wrapper;
  wrapper = std::make_unique<OnnxWrapper>(model_file);
  wrapper->initialize();
  const int obsDim = wrapper->get_obsDim();

  // No need to print with high precision
  std::cout << std::setprecision(3);

  // Run one inference
  Vxf input = Vxf::Ones(obsDim);
  Vxf A = wrapper->run(input);
  std::cout << "Output for obs vector full of 1s:" << std::endl
            << A.transpose() << std::endl;

  input = Vxf::Zero(obsDim);
  A = wrapper->run(input);
  std::cout << "Output for obs vector full of 0s:" << std::endl
            << A.transpose() << std::endl;

  // Time the average computation time for inference
  auto t1 = high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    wrapper->run(Vxf::Zero(obsDim));
  }
  auto t2 = high_resolution_clock::now();

  // Getting number of microseconds as a double.
  duration<double, std::micro> ms_double = t2 - t1;

  std::cout << "Average computation time: " << ms_double.count() / 100
            << " us\n";

  return 0;

  // TESTING LINK WITH THE JOYSTICK

  using namespace std::chrono_literals;

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