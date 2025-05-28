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

#include "OnnxWrapper.hpp"
#include "Types.h"

constexpr float pi_v = 3.14159265358979323846;

class Interface {
public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Interface();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Interface(){};

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initializer
  ///
  /// \param[in] model_file Path to the .onnx model file that contains policy
  /// parameters \param[in] q_ref Reference joint configuration around which to
  /// apply the actions \param[in] dt Control time step
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(std::basic_string<ORTCHAR_T> model_file, const Vxf &q_ref,
                  float dt);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Forward pass
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vxf forward();
  Vxf forward_ManiSkill(); // Forward pass with ManiSkill policy

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run observation network and refresh observation vector
  ///
  /// \param[in] pos Joint positions
  /// \param[in] vel Joint velocities
  /// \param[in] rpy Base orientation (RPY angles)
  /// \param[in] ori Base orientation (quaternion)
  /// \param[in] gyro Base angular velocities
  /// \param[in] cmd Command vector
  /// \param[in] time Elapsed time to compute limb phases
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_observation(const Vector19 &pos, const Vector19 &vel,
                          const Vector19 &tau, const Vector3 &rpy,
                          const Vector4 &ori, const Vector3 &gyro,
                          const Vector6 &cmd, float time);
  void update_observation_with_clock(const Vector19 &pos, const Vector19 &vel,
                                     const Vector19 &tau, const Vector3 &rpy,
                                     const Vector4 &ori, const Vector3 &gyro,
                                     const Vector6 &cmd, float time);
  void update_observation_ManiSkill(const Vector19 &pos, const Vector19 &vel,
                                    const Vector19 &tau, const Vector3 &rpy,
                                    const Vector4 &ori, const Vector3 &gyro,
                                    const Vector6 &cmd, float time, float loco_mode);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Refresh history vector based on previously computed observation
  /// vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_history();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Translate the body rotation into a rotated gravity vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void transformBodyQuat();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Reorder an observation for ManiSkill joint order
  ///
  /// \param[in] v The observation vector to be reordered
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector19 reorder_obs(const Vector19 &v);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Reorder an action from a ManiSkill policy
  ///
  /// \param[in] v The action vector to be reordered
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vxf reorder_act(const Vxf &v);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Return the computation time to run the observation and control
  /// networks
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  float get_computation_time() {
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_end_ - t_start_)
            .count());
  }

  int get_obsDim() { return policy_->get_obsDim(); }
  int get_actDim() { return policy_->get_actDim(); }

  // Control policy
  std::shared_ptr<OnnxWrapper> policy_;

  // Misc
  Vector3 vel_command_ = Vector3::Zero();
  Vxf pTarget_, q_ref_, obs_, actorObs_, studentObs_, historyObs_,
      historyTempObs_, latentOut_, actions_;
  Vxf last_actions_; // , last_dof_pos_, last_dof_vel_;
  int obsDim_, actDim_, historyLength_, historySamples_, historyStep_, iter_;
  float dt_;
  std::chrono::time_point<std::chrono::steady_clock> t_start_;
  std::chrono::time_point<std::chrono::steady_clock> t_end_;

  // Related to leg phases
  Vector2 phases_freq_;
  Eigen::Array<float, 2, 1> phases_;

  // Related to orientation
  Vector3 _bodyOri, _bodyAngularVel, _gravityVec, _qa, _qb, _qc, _qvec;
  Vector4 _bodyQuat;
};

Interface::Interface() {

  // History is not used for now so we can hardcode 1s
  historySamples_ = 1;
  historyStep_ = 1;
  historyLength_ = 1 + historyStep_ * (historySamples_ - 1);
  iter_ = 0;
}

void Interface::initialize(std::basic_string<ORTCHAR_T> model_file,
                           const Vxf &q_ref, float dt) {

  // Initialize ONNX framework
  policy_ = std::make_shared<OnnxWrapper>(model_file);
  policy_->initialize();

  // Retrieve info about network
  obsDim_ = policy_->get_obsDim();
  actDim_ = policy_->get_actDim();

  std::cout << "Network parameters: " << std::endl;
  std::cout << "obsDim: " << obsDim_ << " | actDim: " << actDim_ << std::endl;

  // Initialize some tensors
  obs_ = Vxf::Zero(obsDim_);
  actorObs_ = Vxf::Zero(obsDim_);
  studentObs_ = Vxf::Zero(obsDim_ * historySamples_);
  historyObs_ = Vxf::Zero(obsDim_ * historyLength_);
  historyTempObs_ = Vxf::Zero(obsDim_ * historyLength_);
  actions_ = Vxf::Zero(actDim_);
  last_actions_ = Vxf::Zero(actDim_);
  pTarget_ = Vxf::Zero(actDim_);
  /*last_actions_ = Eigen::MatrixXf::Zero(nJoints, 6);
  last_dof_pos_ = Eigen::MatrixXf::Zero(nJoints, 6);
  last_dof_vel_ = Eigen::MatrixXf::Zero(nJoints, 6);*/

  // Reference position around which to apply the actions
  q_ref_ = q_ref;

  // Initial phases
  phases_freq_.setZero();
  phases_ << 0.0, pi_v;

  // Related to orientation
  _gravityVec << 0.0, 0.0, 1.0;
  _qa.setZero();
  _qb.setZero();
  _qc.setZero();
  _qvec.setZero();

  // Initial times
  dt_ = dt;
  t_start_ = std::chrono::steady_clock::now();
  t_end_ = std::chrono::steady_clock::now();
}

Vxf Interface::forward() {

  // Compute policy actions
  actions_ = policy_->run(obs_);

  // Target joint positions based on scaled actions
  assert(q_ref_.rows() == actDim_);
  pTarget_ = q_ref_ + (0.5 * actions_ + 0.5 * last_actions_);
  assert(pTarget_.rows() == actDim_);

  // Remember actions for next step
  last_actions_ = actions_;

  // Log time
  t_end_ = std::chrono::steady_clock::now();

  return pTarget_;
}

void Interface::update_observation_with_clock(
    const Vector19 &pos, const Vector19 &vel, const Vector19 &tau,
    const Vector3 &rpy, const Vector4 &ori, const Vector3 &gyro,
    const Vector6 &cmd, float time) {
  // Log time
  t_start_ = std::chrono::steady_clock::now();

  const float total_duration = 1.00;
  float phase = 2 * pi_v * (time / total_duration);

  float roll = rpy(0);
  float pitch = rpy(1);
  Vector3 base_ang_vel = gyro;

  // Filling observation vector
  obs_ << roll,
          pitch,
          base_ang_vel,
          pos.head(10),
          vel.head(10),
          tau.head(10),
          std::sin(phase),
          std::cos(phase),
          0, 1, 0, 0, 0, 0;
  assert(obs_.rows() == obsDim_);

  // Iteration counter
  iter_++;
}

void Interface::update_observation(const Vector19 &pos, const Vector19 &vel,
                                   const Vector19 &tau, const Vector3 &rpy,
                                   const Vector4 &ori, const Vector3 &gyro,
                                   const Vector6 &cmd, float time) {
  // Log time
  t_start_ = std::chrono::steady_clock::now();

  float roll = rpy(0);
  float pitch = rpy(1);
  Vector3 base_ang_vel = gyro;

  // Filling observation vector
  obs_ << roll, pitch, base_ang_vel, pos.head(10), vel.head(10), tau.head(10);
  assert(obs_.rows() == obsDim_);

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
  // Discard the last observation sample in history and insert the latest one at
  // the beginning
  historyTempObs_ = historyObs_;
  if (historyLength_ > 1) {
    historyObs_.tail(obsDim_ * (historyLength_ - 1)) =
        historyTempObs_.head(obsDim_ * (historyLength_ - 1));
  }

  // Insert new observations into history
  historyObs_.head(obsDim_) = obs_; // observationScaler_.scale(obs_);

  // Fill observation vector for student by extracting the right samples from
  // the observation history for (int i = 0; i < historySamples_; i++) {
  //   studentObs_.block(i * obsDim_, 0, obsDim_, 1) = historyObs_.block(i *
  //   historyStep_ * obsDim_, 0, obsDim_, 1);
  // }
}

void Interface::transformBodyQuat() {
  // Body QUAT and gravity vector of 0 , 0, +1
  float q_w = _bodyQuat[3];
  _qvec = _bodyQuat.head(3);
  _qa = _gravityVec * (2. * q_w * q_w - 1.);
  _qb = _qvec.cross(_gravityVec) * q_w * 2.0;
  _qc = _qvec * (_qvec.transpose() * _gravityVec) * 2.0;
  _bodyOri = _qa - _qb + _qc;
}

////
// FOR MANISKILL
////

Vector19 Interface::reorder_obs(const Vector19 &v) {

  // From URDF order to ManiSkill order
  Vector19 out = Vector19::Zero();
  int idx[19] = {0,  5, 10, 1,  6,  11, 15, 2,  7, 12,
                 16, 3, 8,  13, 17, 4,  9,  14, 18};
  for (int i = 0; i < 19; i++) {
    out[i] = v[idx[i]];
  }
  return out;
}

Vxf Interface::reorder_act(const Vxf &v) {

  // From ManiSkill order to URDF order
  Vxf out = Vxf::Zero(19);
  int idx[19] = {0, 3, 7, 11, 15, 1, 4, 8, 12, 16, // Legs
                 2, // Torso
                 5, 9, 13, 17, 6, 10, 14, 18};  // Arms
  for (int i = 0; i < 19; i++) {
    out[i] = v[idx[i]];
  }
  return out;
}

Vxf Interface::forward_ManiSkill() {

  // Compute policy actions
  actions_ = policy_->run(obs_);

  // Force torso to 0
  /*actions_[2] = 0.0;
  for (int i = 10; i < policy_out_.rows(); ++i) {
    actions_[i] = 0.0;
  }*/

  // Force torso and arms to 0
  std::array<int, 9> idx = {2, 5, 6, 9, 10, 13, 14, 17, 18};
  for (size_t i = 0; i < idx.size(); ++i) {
    actions_[idx[i]] = 0.0;
  }
  actions_[2] = 0.0;

  // Force arms to 0 except pitch
  /* actions_[9] = 0.0;
  actions_[10] = 0.0;
  actions_[13] = 0.0;
  actions_[14] = 0.0;
  actions_[17] = 0.0;
  actions_[18] = 0.0; */

  // Target joint positions based on scaled actions
  assert(q_ref_.rows() == reorder_act(actions_).rows());
  pTarget_ = q_ref_ + 0.75 * reorder_act(actions_) + 0.25 * last_actions_ ;

  assert(pTarget_.rows() == q_ref_.rows());

  last_actions_.col(0) = reorder_act(actions_);

  // Log time
  t_end_ = std::chrono::steady_clock::now();

  return pTarget_;
}

void Interface::update_observation_ManiSkill(
    const Vector19 &pos, const Vector19 &vel, const Vector19 &tau,
    const Vector3 &rpy, const Vector4 &ori, const Vector3 &gyro,
    const Vector6 &cmd, float time, float loco_mode) {
  // Log time
  t_start_ = std::chrono::steady_clock::now();

  // Projected gravity based on orientation state
  // _bodyQuat = ori;
  // transformBodyQuat(); // this update _bodyOri
  // Vector3 projected_gravity = _bodyOri;

  float roll = rpy(0);
  float pitch = rpy(1);

  float phase = 2 * pi_v * 1.2 * time;

  Vector3 base_ang_vel = gyro;

  
  Vector10 pos_lower, vel_lower, act_lower;
  Vector19 reordered_pos = reorder_obs(pos);
  Vector19 reordered_vel = reorder_obs(vel);

  std::array<int, 10> idx = {0, 1, 3, 4, 7, 8, 11, 12, 15, 16};
  for (size_t i = 0; i < idx.size(); ++i) {
    pos_lower[i] = reordered_pos[idx[i]];
    vel_lower[i] = reordered_vel[idx[i]];
    act_lower[i] = actions_[idx[i]];
  }

  // Filling observation vector
  obs_ << -base_ang_vel(0),
          -base_ang_vel(1),
          base_ang_vel(2),
          roll,
          pitch,
          pos_lower,
          vel_lower,
          act_lower,
          /*reorder_obs(pos),
          reorder_obs(vel),
          actions_,*/
          loco_mode,
          cmd(0),
          cmd(1),
          cmd(5),
          std::cos(phase),
          std::sin(phase),
          Vxf::Zero(4); // Unused by actor but was there for critic
          //Vxf::Zero(319); // Unused by actor

  assert(obs_.rows() == obsDim_);

  /*std::cout << "== " << time << " ==" << std::endl;
  std::cout << obs_.transpose() << std::endl;
  std::cout << "== " << std::endl;
  std::cout << base_ang_vel.transpose() << std::endl;
  std::cout << roll << std::endl;
  std::cout << pitch << std::endl;
  std::cout << reorder_obs(pos).transpose() << std::endl;
  std::cout << reorder_obs(vel).transpose() << std::endl;
  std::cout << actions_.transpose() << std::endl;
  std::cout << cmd(0) << std::endl;
  std::cout << cmd(1) << std::endl;
  std::cout << cmd(5) << std::endl;
  std::cout << std::cos(phase) << std::endl;
  std::cout << std::sin(phase) << std::endl;
  std::cout << "== == == ==" << std::endl;*/

  // Iteration counter
  iter_++;
}