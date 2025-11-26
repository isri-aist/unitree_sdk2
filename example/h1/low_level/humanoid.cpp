#include "humanoid.hpp"

int main(int argc, char const *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " networkInterface pathToOnnxModel scaling" << std::endl;
    exit(-1);
  }

  HumanoidExample example(argv[1], argv[2], argv[3]);
  while (1) {
    sleep(10);
  }
  return 0;
}