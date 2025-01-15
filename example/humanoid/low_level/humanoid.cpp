#include "humanoid.hpp"

int main(int argc, char const *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " networkInterface pathToOnnxModel" << std::endl;
    exit(-1);
  }

  HumanoidExample example(argv[1], argv[2]);
  while (1) {
    sleep(10);
  }
  return 0;
}
