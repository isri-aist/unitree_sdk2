add_executable(h1_loco_client high_level/h1_loco_client_example.cpp)
target_link_libraries(h1_loco_client unitree_sdk2)

add_executable(g1_loco_client high_level/g1_loco_client_example.cpp)
target_link_libraries(g1_loco_client unitree_sdk2)

include_directories(/home/paleziart/git/sdk/unitree_sdk2-rohan/NatNetSDK-4.1/include)
link_directories(/home/paleziart/git/sdk/unitree_sdk2-rohan/NatNetSDK-4.1/lib)
add_executable(h1_low_level_example low_level/humanoid.cpp low_level/Joystick.cpp)
target_link_libraries(h1_low_level_example unitree_sdk2 onnxruntime NatNet)

add_executable(h1_test_example low_level/debug_humanoid.cpp low_level/Joystick.cpp)
target_link_libraries(h1_test_example unitree_sdk2 onnxruntime)

