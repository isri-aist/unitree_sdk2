#include "Joystick.hpp"

Joystick::Joystick()
    : A3_(Vector6::Zero()), A2_(Vector6::Zero()), v_ref_(Vector6::Zero()),
      v_gp_(Vector6::Zero()) {}

Joystick::~Joystick() {
  if (js != -1) {
    close(js);
  }
}

void Joystick::initialize(double dt) {
  dt_ = dt;

  // Gamepad initialisation
  if (!predefined) {
    device = "/dev/input/js0";
    js = open(device, O_RDONLY | O_NONBLOCK);
    if (js == -1) {
      perror("Could not open joystick");
    }
  }
}

void Joystick::handle_v_switch(int k) {
  int i = 1;
  while (i < k_switch.size() && k_switch(i) <= k) {
    i++;
  }
  if (i != k_switch.size()) {
    double ev = k - k_switch(i - 1);
    double t1 = k_switch(i) - k_switch(i - 1);
    A3_ = 2 * (v_switch.col(i - 1) - v_switch.col(i)) / pow(t1, 3);
    A2_ = (-3.0 / 2.0) * t1 * A3_;
    v_ref_ = v_switch.col(i - 1) + A2_ * pow(ev, 2) + A3_ * pow(ev, 3);
  }
}

void Joystick::update_v_ref(int k) {
  if (predefined) {
    if (analysis) {
      handle_v_switch(k);
    } else {
      update_v_ref_predefined(k);
    }
  } else {
    update_v_ref_gamepad(k);
  }
}

int Joystick::read_event(int fd, struct js_event *event) {
  ssize_t bytes;
  bytes = read(fd, event, sizeof(*event));
  if (bytes == sizeof(*event))
    return 0;
  /* Error, could not read full event. */
  return -1;
}

void Joystick::update_v_ref_gamepad(int k) {
  // Read information from gamepad client
  if (read_event(js, &event) == 0) {
    if (event.type == JS_EVENT_BUTTON) {
      switch (event.number) {
      case 9:
        gamepad.start = event.value;
        break;
      case 8:
        gamepad.select = event.value;
        break;
      case 0:
        gamepad.cross = event.value;
        break;
      case 1:
        gamepad.circle = event.value;
        break;
      case 2:
        gamepad.triangle = event.value;
        break;
      case 3:
        gamepad.square = event.value;
        break;
      case 4:
        gamepad.L1 = event.value;
        break;
      case 5:
        gamepad.R1 = event.value;
        break;
      }
    } else if (event.type == JS_EVENT_AXIS) {
      if (event.number == 0)
        gamepad.v_y = -event.value / 32767.0;
      else if (event.number == 1)
        gamepad.v_x = -event.value / 32767.0;
      else if (event.number == 4)
        gamepad.v_z = -event.value / 32767.0;
      else if (event.number == 3)
        gamepad.w_yaw = -event.value / 32767.0;
    }
  }

  // Retrieve data from gamepad for velocity
  double vX = gamepad.v_x * vXScale;
  double vY = gamepad.v_y * vYScale;
  double vYaw = gamepad.w_yaw * vYawScale;
  v_gp_ << vX, vY, 0.0, 0.0, 0.0, vYaw;

  // Dead zone to avoid gamepad noise
  double dead_zone = 0.004;
  for (int i = 0; i < 6; i++) {
    if (v_gp_(i, 0) > -dead_zone && v_gp_(i, 0) < dead_zone) {
      v_gp_(i, 0) = 0.0;
    }
  }

  // Switch to safety controller if the select key is pressed
  if (gamepad.select == 1) {
    stop_ = true;
  }
  if (gamepad.start == 1) {
    start_ = true;
  }

  // Joystick code
  joystick_code_ = 0;

  // Low pass filter to slow down the changes of velocity when moving the
  // joysticks
  v_ref_ = gp_alpha_vel * v_gp_ + (1 - gp_alpha_vel) * v_ref_;
}

void Joystick::update_v_ref_predefined(int k) {
  /*if (k == 0) {
    v_swich = params_->v_switch;
    k_switch = (params_->t_switch / dt_).cast<int>();
  }*/

  // Polynomial interpolation to generate the velocity profile
  handle_v_switch(k);
}

void Joystick::update_for_analysis(Vector6 des_vel_analysis, int N_analysis,
                                   int N_steady) {
  analysis = true;
  double v_step = 0.05;                  // m/s
  double v_max = des_vel_analysis(0, 0); // m/s
  int n_steps = static_cast<int>(std::round(v_max / v_step));
  int N_start =
      static_cast<int>(std::round(1.0 / dt_)); // Wait 1s before starting
  int N_slope = static_cast<int>(
      std::round(1.0 / dt_)); // Acceleration between steps last 1s
  int N_still =
      static_cast<int>(std::round(3.0 / dt_)); // Steady velocity phases last 5s

  // Set dimensions of arrays
  k_switch = Eigen::Matrix<int, 1, Eigen::Dynamic>::Zero(1, 2 * (n_steps + 1));
  v_switch = Mxf::Zero(6, 2 * (n_steps + 1));

  // Fill them
  k_switch(0, 0) = 0;
  k_switch(0, 1) = N_start;
  for (int i = 1; i <= n_steps; i++) {
    k_switch(0, 2 * i) = k_switch(0, 2 * i - 1) + N_slope;
    k_switch(0, 2 * i + 1) = k_switch(0, 2 * i) + N_still;
    v_switch(0, 2 * i) = std::cos(des_vel_analysis(3, 0)) * i * v_step;
    v_switch(0, 2 * i + 1) = std::cos(des_vel_analysis(3, 0)) * i * v_step;
    if (des_vel_analysis(1, 0) != 0.0) {
      v_switch(1, 2 * i) = std::sin(des_vel_analysis(3, 0)) * i * v_step;
      v_switch(1, 2 * i + 1) = std::sin(des_vel_analysis(3, 0)) * i * v_step;
    } else {
      v_switch(5, 2 * i) = std::sin(des_vel_analysis(3, 0)) * i * v_step;
      v_switch(5, 2 * i + 1) = std::sin(des_vel_analysis(3, 0)) * i * v_step;
    }
  }
}

Vector6 Joystick::getLastReachedVelocity(int k) {
  int i = 1;
  while ((i < k_switch.cols()) && k_switch(0, i) <= k) {
    i++;
  }
  Vector6 v_reached;
  if ((v_switch.col(i - 1)).isApprox(v_switch.col(i))) {
    if (i - 2 >= 0) {
      v_reached = v_switch.col(i - 2);
    } else {
      v_reached = Vector6::Zero();
    }
  } else {
    v_reached = v_switch.col(i - 1);
  }
  return v_reached;
}
