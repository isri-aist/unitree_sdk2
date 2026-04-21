#include "humanoid.hpp"

int main(int argc, char const *argv[]) {
  if (argc != 5) {
    std::cout << "Usage: " << argv[0] << " networkInterface pathToOnnxModel1 pathToOnnxModel2 pathToOnnxModel3" << std::endl;
    exit(-1);
  }

  HumanoidExample example(argv[1], argv[2], argv[3], argv[4]);
  while (1) {
    sleep(10);
  }
  return 0;
}
