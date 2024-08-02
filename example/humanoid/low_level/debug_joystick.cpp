#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string>
#include <thread>
#include "Joystick.hpp"
#include "Joystick.cpp"

int main() {

  using namespace std::chrono_literals;

  Joystick joy;
  joy.initialize(0.001);

  while (true) {
    joy.update_v_ref();
    std::this_thread::sleep_for(1ms);

    std::cout << "----" << std::endl;
    std::cout << joy.getVRef() << std::endl;
    std::cout << joy.getStart() << std::endl;
    std::cout << joy.getStop() << std::endl;
  }

  return 0;
}