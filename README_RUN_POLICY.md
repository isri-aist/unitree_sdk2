# How to deploy a policy on H1

Quick guide on how to run an ONNX policy on H1 in pure C++ using the unitree sdk 2.


## Installation

- Create a working directory where you will store all resources.
- In that directory, create an `install` directory that will store compiled C++ libraries required for unitree sdk2
- Download the sources of Eigen 3.4.0, build and install.
```
cd eigen-3.4.0
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/your_path/install
make install -sj2
```
- Download the sources of fmtlib, build and install.
```
git clone --recursive https://github.com/fmtlib/fmt.git
cd fmt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/your_path/install
make install -sj2
```
- Download the sources of fmtlib, build and install.
```
git clone --recursive https://github.com/seleznevae/libfort.git
cd libfort
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/your_path/install
make install -sj2
```
- Download a compiled release of ONNX Runtime (tested with 1.21, I think 1.22 had an issue). Get it from https://github.com/microsoft/onnxruntime/releases/ and store the unpacked folder in the working directory (not in install). Inside the onnxruntime folder you should see an include and a lib folders. It will likely be named onnxruntime-linux-x64-1.21.0 or something like that, you can shorten it to onnxruntime.
- Download the unitree sdk2 from AIST github. The devel-PA-new branch has additional features on top of unitree sdk2 to be able to run a policy, have a joystick, have logging, security checks.
```
git clone https://github.com/isri-aist/unitree_sdk2.git
git checkout devel-PA-new
```
- Modify the CMakeLists.txt to point to your install folder.
```
target_include_directories(unitree_sdk2 INTERFACE /home/h1user/paleziart/install/lib)
target_include_directories(unitree_sdk2 INTERFACE /home/h1user/paleziart/install/include)
```
- Modify the ONNXRUNTIME_ROOTDIR directory to point to your onnxruntime folder.
```
set(ONNXRUNTIME_ROOTDIR "/home/h1user/paleziart/onnxruntime")
```
- Create a build directory for unitree sdk2 and make install:
```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/your_path/install
make install -sj2
```

## How is the code organized?

Code is stored in `unitree_sdk2/example/h1/low_level`

- `humanoid.hpp`: The main control loop, state-machine, logging, plotting.
- `Interface.hpp`: Where you will have do make update so that the observation vector contains what the policy is expecting. There is also a history mechanism that you can use to send an history of observations.
- `logging.hpp`: Logging utilities if you want to expand what is logged.
- `Joystick.hpp`: If you need to change the Joystick max values, low pass filtering.
- `OnnxWrapper.hpp`: If you need to debug the way networks are loaded and how inference is done. Be careful if you modify this one because it uses specific pointers and memory mapping for ONNX.


## How to enable/disable the gamepad

In humanoid.hpp you can change this line `#define USE_JOYSTICK true`. If you set it to True, the velocity command of the robot will come from the gamepad you connected to the computer. If you set it to False, you will have to define manually the changes of velocity commands in `HumanoidExample::Control()`.

## How to validate the code and the policy in simulation

You can use unitree_mujoco to validate your pipeline before deploying on the real robot. This is important in case you changed something and made a mistake in the order of observations, or if an array is not properly initialized and leads to infinite values. What unitree_mujoco does is to open a mujoco simulation that listens to the exact same interface than unitree_sdk2. It is a way to validate that you are processing correctly the sensor data coming from the robot, that your inference is properly done and that you are sending the right commands. If the robot walks well in unitree_mujoco it is a good sign that there is no mistake in the code. Although there is no guarantee it will actually behave well on the real robot because of the sim-to-real gap (but at least the robot will not go crazy because you swapped two quantities when modifying the observations).

If it does not behave well in simulation:
- check that the order of observations is correct in Interface.hpp
- check that you have the right PD gains in humanoid.hpp
- check that the history length is correct
- check that you properly scale the observations (if the normalization is not done directly by the ONNX inference) or the actions (if you multiply the actions by a given value).
- check that you have the same default joint configuration than what you trained with.
- check that you are using data in the right frames

For the full installation README, check https://github.com/unitreerobotics/unitree_mujoco

Install the dependencies:
```
sudo apt install libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev
```

You already have installed unitree_sdk2 in your `install` folder, so we can skip what they say about `/opt/unitree_robotics`

Download the latest mujoco release, and extract it to a `mujoco` folder in the working directory.

Make a symbolic link between mujoco and unitree_mujoco
```
cd unitree_mujoco/simulate/
ln -s ~/.mujoco/mujoco-3.3.6 mujoco
```

Compile unitree_mujoco:
```
cd unitree_mujoco/simulate
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/your_path/install
make -sj4
```

Run:
```
./unitree_mujoco -r h1 -s scene_terrain.xml
```
You should see the mujoco simulator with the H1 robot loaded.

You can now launch the controller of unitree_sdk2 in another terminal (get into the build folder of unitree sdk2). 
```
./bin/h1_low_level_example dummyName ~/Documents/pa-policies/2025-09-18_18-55-44_final_ckpt 1.0
```
which abide by the format
```
./bin/h1_low_level_example NAME_OF_INTERFACE PATH_TO_FOLDER_WITH_ONNX_POLICIES KP_GAINS_SCALING
```

It should start in Waiting mode and the robot will get in the default configuration. You can click the "Reset" button on the left GUI in the simulation window to put the robot back on its feet. Then press Enter in the controller terminal to switch to "Waiting on the floor" (intended when using the robot with a crane), and then finally press Enter again to run with policy inference. You should be able to move around with the joystick to confirm the behavior of your policy. When you kill the controller with Ctrl+C you do not need to close the mujoco simulation, it can keep running in the background while you fix your unitree_sdk2 code.

## Starting the real robot

- Before turning on the robot suspended on the crane, raise the tip of the feet upwards to reach the mechanical limit.
- Turn the yaw DoF of the arm until reaching the mechanical limit (the power cable should wrap around the arm) 
- Roughly align the arm and the elbows on the side of the body. The starting position of the arm joints will be considered
the 0 position for the controller so if you don't align them well the arm joints will have a constant offset.
- To start, for both batteries at the same time, press the button once shortly, then release and keep pressed again.
- For the unitree gamepad, same for turning it on, short press, then release, then keep pressing.

## Initializing the robot

- Wait for 30-60 seconds to let the internal computer the time to boot properly otherwise it might suddenly fails when launching the control.
- Press L2+R2 on the controller.
- Press L2+A, wait until it reaches the default pose. Then L2+A again and wait. Then again L2+A and it should end up close from the default pose.
- Press L2+B to stop the controller.

## Running the controller

- Turn on the PS5 controller with "RL locomotion team" on the back
- Check that the wired connection of Ubuntu is set to "H1 control" so that the robot receives the commands. If the connection does not exist, create it by setting a manual IPv4 connection with adress 192.168.123.222 and mask 255.255.255.0
- cd ~/paleziart/unitree_sdk2/build
- make -sj2 (should already be built from your simulation check)
- `./bin/h1_low_level_example enx04ab18ffbad8 ~/Documents/pa-policies/2025-09-18_18-55-44_final_ckpt 0.0`. For the name of the interface, check the ethernet name with `ifconfig`.
- We have set Kp scaling to "0.0" to run the policy without actually controlling the robot. This is in case you have a segfault or something wrong happening when deploying on the real robot. 
- The robot should go into the default position, you can  press Enter a few times until the terminal displays "PD gains transition" for the status.
- In case of emergency, you can press the X button of the gamepad to switch to damping mode to smoothly kill the controller.
In real emergency you can also Ctrl+C to kill all control.
- When you press the "options" button next to square and triangle (sometimes called "Start" instead of "options"), the controller will switch to the policy network. Since Kp scaling is 0.0 the robot should look like its not controlled at all, but you can check what the policy is sending with what is displayed in the terminal.
- If nothing seems broken, kill the controller, and launch with `./bin/h1_low_level_example enx04ab18ffbad8 ~/Documents/pa-policies/2025-09-18_18-55-44_final_ckpt 0.1` to test the policy with only 10% of the actual gains. Lower the crane until the feet touch the ground, but keep the rope tense to hold the robot. The policy at 10% will not be enough to generate the torques required to hold the full weight of the robot. You can use the Left stick to control the forward and lateral velocity command and Right stick to control turning left/right in yaw. You should see the feet moving a bit.
- If everything went well, launch again with 0.25 scaling, then 0.5, 0.75 and finally 1.0. At 1.0 you should be able to fully lower the crane since the policy is trained to hold the full weight of the robot. If you see that the robot is hitting the floor very hard or if the robot starts shaking even when the scaling is 0.5 or 0.75, it might be a good idea to stop here and check that you have all the right settings before breaking the robot!!

## The robot suddenly stopped?!

- It happens for some reason when the robot runs during several minutes without stopping. Might be a buffer reason?
- You will likely have to shut down the robot and restart everything because when that happens the robot becomes unresponsive.

## To show how smooth the joint are.

- When the robot is not controlled, you can L2+B to disable the in-built damping and feel how tranparent the joints are.
- Press L2+B again to turn it back on.
- Don't play to much with the joints because it creates back-currents when you move the joints manually. If the electronics
is well designed there are resistances made to dissipate the rise of voltage in the capacitors, but better be safe than sorry. 

## Side note

For a good omnidirectional walking policy on flat ground, get on the following commit, compile then run the policy. Validate first in simulation to be sure.
 
66d6d4a (HEAD -> devel-PA) Various changes for threads, control logic, gains, security checks
./bin/h1_low_level_example enx04ab18ffbad8 ~/Documents/pa-policies/2025-04-21_19-55-39_final_ckpt.onnx

There is no Kp scaling as argument for this commit.

