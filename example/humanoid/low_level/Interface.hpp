///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Interface class
///
/// \details C++ interface between the control loop and the low-level neural
/// network code
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <iostream>

#include "Types.h"
#include "OnnxWrapper.hpp"

constexpr float pi_v = 3.14159265358979323846;

class Interface {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  /// \param[in] obsDim Size of the observation vector
  /// \param[in] latentDim Size of the latent space vector
  /// \param[in] nJoints Number of joints (actions)
  /// \param[in] historySamples Number of observations samples provided to the student
  /// \param[in] historyStep Number of control steps between each provided observations samples
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Interface(int obsDim, int latentDim, int nJoints, int historySamples, int historyStep);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Interface();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initializer
  ///
  /// \param[in] model_file Path to the .onnx model file that contains policy parameters
  /// \param[in] q_ref Reference joint configuration around which to apply the actions
  /// \param[in] action_scale Scaling parameters for actions
  /// \param[in] dt Control time step
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(std::basic_string<ORTCHAR_T> model_file, VectorM q_ref, float action_scale, float dt);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Forward pass
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  VectorM forward();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run observation network and refresh observation vector
  ///
  /// \param[in] pos Joint positions
  /// \param[in] vel Joint velocities
  /// \param[in] rpy Base orientation (RPY angles)
  /// \param[in] gyro Base angular velocities
  /// \param[in] cmd Command vector
  /// \param[in] time Elapsed time to compute limb phases
  /// \param[in] hmap Heightmap of the terrain around the robot
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_observation(VectorM pos, VectorM vel, VectorM tau, Vector3 rpy, Vector3 gyro, Vector6 cmd, float time); //, VectorN hmap);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Refresh history vector based on previously computed observation vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_history();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Return the observation vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::VectorXf get_observation() { return obs_; }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Translate the body rotation into a rotated gravity vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void transformBodyQuat();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Return the computation time to run the observation and control
  /// networks
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  float get_computation_time() {
    return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(t_end_ - t_start_).count());
  }

  // Control policy
  OnnxWrapper *policy_;

  // Observation scaler
   //Scaler observationScaler_;

  // Encoder policy
  // MLP encoderModel_;
  const bool useEncoder = false;

  // Encoder output scaler
  // Scaler encoderScaler_;

  // scales
  double _scaleAction, _hipReduction, _scaleQ, _scaleQd, _scaleLinVel, _scaleAngVel, _scaleHeights;
  Eigen::Vector3f _scaleCommand;

  // Misc
  VectorM pTarget_;
  VectorM q_ref_;
  Vector3 vel_command_;
  int obsDim_, latentDim_, nJoints_, historyLength_, historySamples_, historyStep_, iter_;
  Eigen::VectorXf obs_, actorObs_, studentObs_, historyObs_, historyTempObs_, latentOut_, actions_;
  Eigen::MatrixXf last_actions_; // , last_dof_pos_, last_dof_vel_;
  Eigen::Matrix<float, 3, 1> _bodyOri;
  Eigen::Matrix<float, 3, 1> _bodyAngularVel;
  Eigen::Matrix<float, 4, 1> _bodyQuat;
  VectorM bound_pi_;
  float dt_;
  std::chrono::time_point<std::chrono::steady_clock> t_start_;
  std::chrono::time_point<std::chrono::steady_clock> t_end_;

  // Related to leg phases
  Vector2 phases_freq_;
  Eigen::Array<float, 2, 1> phases_;

  // orientation
  Eigen::Vector3f _gravityVec, _qa, _qb, _qc, _qvec;
  float q_w = 0.0;
};

Interface::Interface(int obsDim, int latentDim, int nJoints, int historySamples, int historyStep) {
  /*
  policy_ = MLP(std::vector<int>({512, 256, 128}), obsDim + latentDim, nJoints);
  if (useEncoder) {
    encoderModel_ = MLP(std::vector<int>({128, 64}), obsDim * historySamples, latentDim);
    encoderScaler_ = Scaler(latentDim);
  }
  observationScaler_ = Scaler(obsDim);
  */
  policy_ = nullptr;
  obsDim_ = obsDim;
  latentDim_ = latentDim;
  nJoints_ = nJoints;
  historySamples_ = historySamples;
  historyStep_ = historyStep;
  historyLength_ = 1 + historyStep * (historySamples - 1);
  iter_ = 0;
  obs_ = Eigen::VectorXf::Zero(obsDim);
  actorObs_ = Eigen::VectorXf::Zero(obsDim + latentDim);
  studentObs_ = Eigen::VectorXf::Zero(obsDim * historySamples);
  historyObs_ = Eigen::VectorXf::Zero(obsDim * historyLength_);
  historyTempObs_ = Eigen::VectorXf::Zero(obsDim * historyLength_);
  latentOut_ = Eigen::VectorXf::Zero(latentDim);
  actions_ = Eigen::VectorXf::Zero(nJoints);
  last_actions_ = Eigen::VectorXf::Zero(nJoints);
  /*last_actions_ = Eigen::MatrixXf::Zero(nJoints, 6);
  last_dof_pos_ = Eigen::MatrixXf::Zero(nJoints, 6);
  last_dof_vel_ = Eigen::MatrixXf::Zero(nJoints, 6);*/

  std::cout << "Network parameters: " << std::endl;
  std::cout << "obsDim: " << obsDim_ << " | latentDim: " << latentDim_ << " | nJoints: " << nJoints_
            << " | historyLength: " << historyLength_ << std::endl;
}

Interface::~Interface() {
  delete policy_;
  policy_ = nullptr;
}

void Interface::initialize(std::basic_string<ORTCHAR_T> model_file, VectorM q_ref, float action_scale, float dt) {

  // Initialize ONNX framework
  policy_ = new OnnxWrapper(model_file);
  policy_->initialize();

  /*
  policy_.updateParamFromTxt(polDirName + "actor.txt");
  if (useEncoder) {
    encoderModel_.updateParamFromTxt(polDirName + "encoder.txt");
    encoderScaler_.updateRunningMeanFromTxt(polDirName + "running_mean_latent.txt");
    encoderScaler_.updateRunningVarFromTxt(polDirName + "running_var_latent.txt");
  }
  observationScaler_.updateRunningMeanFromTxt(polDirName + "running_mean.txt");
  observationScaler_.updateRunningVarFromTxt(polDirName + "running_var.txt");
  */

  // Action scale
  _scaleAction = action_scale;
  _hipReduction = 1.0;

  // Obs scales
  _scaleQ = 1.0;
  _scaleQd = 1.0;
  _scaleLinVel = 1.0;
  _scaleAngVel = 1.0;
  _scaleHeights = 1.0;
  _scaleCommand << _scaleLinVel, _scaleLinVel, _scaleAngVel;

  // Velocity command
  vel_command_ = Vector3(0.0f, 0.0f, 0.0f);

  // Reference position around which to apply the actions
  q_ref_ = q_ref;

  // Initial phases
  phases_freq_.setZero();
  phases_ << 0.0, pi_v;

  // SetZero
  pTarget_.setZero();
  obs_.setZero();
  historyObs_.setZero();
  latentOut_.setZero();
  _gravityVec << 0.0, 0.0, -1.;
  _qa.setZero();
  _qb.setZero();
  _qc.setZero();
  _qvec.setZero();
  bound_pi_ << VectorM::Ones() * pi_v;

  // Initial times
  dt_ = dt;
  t_start_ = std::chrono::steady_clock::now();
  t_end_ = std::chrono::steady_clock::now();
}

VectorM Interface::forward() {
  //if (useEncoder) {
  //  latentOut_ = encoderModel_.forward(studentObs_);

    /* std::cout << "v_latent    " << latentOut_.transpose() << std::endl;
    Vector3 A = encoderScaler_.scale(latentOut_);
    std::cout << "nv_latent    " << A.transpose() << std::endl; */

  //  actorObs_ << historyObs_.head(obsDim_), encoderScaler_.scale(latentOut_);
  //} else {
  //  actorObs_ << historyObs_.head(obsDim_);
  //}
  // std::cout << obsDim_ << std::endl;
  // std::cout << latentDim_ << std::endl;

  // std::cout << "-----" << std::endl;
  // std::cout << "actorObs_ " << std::endl << actorObs_.transpose() << std::endl;
  
  // actions_ = policy_.forward(actorObs_);

  // std::cout << "Actions: " << std::endl << (actions_).transpose() << std::endl;

  // Force arm actions to 0
  // actions_.tail(4) *= 0.0;

  // std::cout << "================" << std::endl;
  // std::cout << actions_ << std::endl;
  // std::cout << "---raw actions: " << std::endl << actions_.transpose() << std::endl;

  /*if (iter_ < 5) {
  // std::cout << "---raw hist: " << historyObs_.transpose() << std::endl;

  std::cout << studentObs_.rows() << 137 * 5 << std::endl;

  std::cout << "---raw student" << std::endl << studentObs_.transpose() << std::endl;
  std::cout << "---raw obs: " << std::endl << obs_.transpose() << std::endl;
  std::cout << "---raw latent: " << latentOut_.transpose() << std::endl;
  std::cout << "---raw actions: " << std::endl << actions_.transpose() << std::endl;
  }*/
  // std::cout << "Actions: " << std::endl << (_scaleAction * actions_).transpose() << std::endl;
  // std::cout << "Q Init: " << std::endl << q_ref_.transpose() << std::endl;

  // actions_ << 0.31365,  0.2679 , -0.57505, -0.31365,  0.2679 , -0.57505, 0.31365,  -0.2679 , 0.57505, -0.31365,
  // -0.2679 , 0.57505;

  // Compute policy actions
  actions_ = policy_->run(obs_); // historyObs_.head(obsDim_));

  // Target joint positions based on scaled actions
  pTarget_ = (0.5 * actions_ + 0.5 * last_actions_) + q_ref_;

  // Remember actions for next step
  last_actions_ = actions_;

  // Log time
  t_end_ = std::chrono::steady_clock::now();

  return pTarget_;
}

void Interface::update_observation(VectorM pos, VectorM vel, VectorM tau, Vector3 rpy, Vector3 gyro, Vector6 cmd, float time) {
  // Log time
  t_start_ = std::chrono::steady_clock::now();

  // Projected gravity based on orientation state
  // _bodyQuat = ori;
  // transformBodyQuat();  // this update _bodyOri
  // Vector3 projected_gravity = _bodyOri;

  // std::cout << _bodyQuat.transpose() << std::endl;
  // std::cout << projected_gravity.transpose() << std::endl;
  
  // Compute limb phase
  /*
  float rem = std::fmod(phases_(0), 2 * pi_v);
  bool refresh = ((rem + 2 * pi_v * phases_freq_(0) * dt_) > 2 * pi_v) || (phases_freq_(0) == 0.0);
  float norm = std::sqrt(cmd(0) * cmd(0) + cmd(1) * cmd(1) + cmd(5) * cmd(5));
  const float deadzone = 0.1;

  phases_ += 2 * pi_v * phases_freq_(0) * dt_;

  if (refresh) {
    phases_freq_(0) = phases_freq_(1);
    if (norm < deadzone) {
      phases_freq_(1) = 0.0;
    } else {
      phases_freq_(1) = 1.25 + 0.5 * (norm - deadzone);
    }
  }
  */

  /*
  const float phase_freq = 1.25;
  Eigen::Array<float, 2, 1> phases;
  phases << 0.0, pi_v;
  phases += 2 * pi_v * phase_freq * time;
  */

  // vel_command_ << cmd(0), cmd(1), cmd(5);

  float roll = rpy(0);
  float pitch = rpy(1);
  Vector3 base_ang_vel = gyro;

  // Filling observation vector
  obs_ << roll,
          pitch,
          base_ang_vel,
          pos,
          vel,
          tau;

  /*obs_ << base_ang_vel * _scaleAngVel,
          vel_command_.cwiseProduct(_scaleCommand),
          // Eigen::cos(phases_),
          // Eigen::sin(phases_),
          projected_gravity,
          pos * _scaleQ,
          vel * _scaleQd,
          actions_ * _scaleAction;*/
  
  // Update the history
  // update_history();

  // Save last actions, joint pos and joint vel
  /* 
  for (int j = 5; j > 0; j--) {
    last_actions_.col(j) = last_actions_.col(j - 1);
    last_dof_pos_.col(j) = last_dof_pos_.col(j - 1);
    last_dof_vel_.col(j) = last_dof_vel_.col(j - 1);
  }
  last_actions_.col(0) = actions_;
  last_dof_pos_.col(0) = pos;
  last_dof_vel_.col(0) = vel;
  */

  // Iteration counter
  iter_++;
}

void Interface::update_history() {
  // Discard the last observation sample in history and insert the latest one at the beginning
  historyTempObs_ = historyObs_;
  if (historyLength_ > 1) {
    historyObs_.tail(obsDim_ * (historyLength_ - 1)) = historyTempObs_.head(obsDim_ * (historyLength_ - 1));
  }

  // Insert new observations into history
  historyObs_.head(obsDim_) = obs_; // observationScaler_.scale(obs_);

  // Fill observation vector for student by extracting the right samples from the observation history
  // for (int i = 0; i < historySamples_; i++) {
  //   studentObs_.block(i * obsDim_, 0, obsDim_, 1) = historyObs_.block(i * historyStep_ * obsDim_, 0, obsDim_, 1);
  // }
}

void Interface::transformBodyQuat() {
  // Body QUAT and gravity vector of 0 , 0, -1
  q_w = _bodyQuat[3];
  _qvec = _bodyQuat.head(3);
  _qa = _gravityVec * (2. * q_w * q_w - 1.);
  _qb = _qvec.cross(_gravityVec) * q_w * 2.0;
  _qc = _qvec * (_qvec.transpose() * _gravityVec) * 2.0;
  _bodyOri = _qa - _qb + _qc;
}
