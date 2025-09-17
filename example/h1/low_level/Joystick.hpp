///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Joystick class
///
/// \details This class handles computations related to the reference velocity of the robot
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JOYSTICK_H_INCLUDED
#define JOYSTICK_H_INCLUDED

#include "Types.h"
#include <chrono>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <linux/joystick.h>

struct gamepad_struct {
  double v_x = 0.0;    // Up/down status of left pad
  double v_y = 0.0;    // Left/right status of left pad
  double v_z = 0.0;    // Up/down status of right pad
  double w_yaw = 0.0;  // Left/right status of right pad
  int start = 0;       // Status of Start button
  int select = 0;      // Status of Select button
  int cross = 0;       // Status of cross button
  int circle = 0;      // Status of circle button
  int triangle = 0;    // Status of triangle button
  int square = 0;      // Status of square button
  int L1 = 0;          // Status of L1 button
  int R1 = 0;          // Status of R1 button
};

class Joystick {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Joystick();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize with given data
  ///
  /// \param[in] dt Time step of the calls to the joystick
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(double dt);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Joystick();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief update the
  ///
  /// \param[in] k Numero of the current loop
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void handle_v_switch(int k);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the status of the joystick, either using polynomial interpolation based on predefined profile or
  /// reading the status of the gamepad
  ///
  /// \param[in] k Numero of the current loop
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_v_ref(int k = 0);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Check if a gamepad event occured and read its data
  ///
  /// \param[in] fd Identifier of the gamepad object
  /// \param[in] event Gamepad event object
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int read_event(int fd, struct js_event* event);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the status of the joystick by reading the status of the gamepad
  ///
  /// \param[in] k Numero of the current loop
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_v_ref_gamepad(int k);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the status of the joystick using polynomial interpolation
  ///
  /// \param[in] k Numero of the current loop
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_v_ref_predefined(int k);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the velocity profile to test the behavior up to a desired velocity
  ///
  /// \param[in] des_vel_analysis Desired velocity that the robot should reach
  /// \param[in] N_analysis Number of controller steps for the increase in velocity (slope)
  /// \param[in] N_steady Number of controller steps at target velocity to be considered stable
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_for_analysis(Vector6 des_vel_analysis, int N_analysis, int N_steady);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Get the last velocity that was reached in the current velocity profile
  ///
  /// \param[in] k Numero of the current loop
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector6 getLastReachedVelocity(int k);

  Vector6 getVRef() { return v_ref_; }
  int getProfileDuration() { return k_switch(0, k_switch.cols() - 1); }
  int getJoystickCode() { return joystick_code_; }
  bool getStop() { return stop_; }
  bool getStart() { return start_; }
  bool getCross() { return gamepad.cross == 1; }
  bool getCircle() { return gamepad.circle == 1; }
  bool getTriangle() { return gamepad.triangle == 1; }
  bool getSquare() { return gamepad.square == 1; }
  bool getL1() { return gamepad.L1 == 1; }
  bool getR1() { return gamepad.R1 == 1; }

 private:

  Vector6 A3_;     // Third order coefficient of the polynomial that generates the velocity profile
  Vector6 A2_;     // Second order coefficient of the polynomial that generates the velocity profile
  Vector6 v_ref_;  // Reference velocity resulting of the polynomial interpolation or after low pass filter
  Vector6 v_gp_;   // Raw velocity reference of the gamepad

  int joystick_code_ = 0;   // Code to trigger gait changes
  bool stop_ = false;       // Flag to stop the controller
  bool start_ = false;      // Flag to start the controller
  bool predefined = false;  // Flag to perform polynomial interpolation or read the gamepad
  bool analysis = false;    // Flag to perform a performance analysis up to a given velocity

  double dt_ = 0.0;  // Time step

  Vxi k_switch;  // Key frames for the polynomial velocity interpolation
  Matrix6N v_switch;  // Target velocity for the key frames

  // How much the gamepad velocity and position is filtered to avoid sharp changes
  const double gp_alpha_vel = 0.0015;  // Low pass filter coefficient for v_ref_ (if gamepad-controlled)

  // Maximum velocity values
  const double vXScale = 0.6;   // Lateral
  const double vYScale = 0.4;   // Forward
  const double vYawScale = 0.8;  // Rotation

  // Gamepad client variables
  struct gamepad_struct gamepad;  // Structure that stores gamepad status
  const char* device;             // Gamepad device object
  int js;                         // Identifier of the gamepad object
  struct js_event event;          // Gamepad event object
};

#endif  // JOYSTICK_H_INCLUDED
