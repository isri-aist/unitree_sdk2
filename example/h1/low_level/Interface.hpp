///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Interface class
///
/// \details C++ interface between the control loop and the low-level neural
/// network code
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <filesystem>
#include <fstream>
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
  // Vxf forward();
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
                                    const Vector6 &cmd, float time);

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

  int get_obsDim() { return policy_actor_->get_obsDim(); }
  int get_actDim() { return policy_actor_->get_actDim(); }

  // Control policy
  std::shared_ptr<OnnxWrapper> policy_actor_, policy_estim_;

  // Misc
  Vector3 vel_command_ = Vector3::Zero();
  Vxf pTarget_, q_ref_, obs_, actorObs_, studentObs_, historyObs_,
      historyTempObs_, latentOut_, actions_, estim_vel_, h_gru_;
  Vxf last_actions_; // , last_dof_pos_, last_dof_vel_;
  Vxf action_scale_;
  Vxf meta_default_qref_;
  bool has_meta_default_qref_;
  int obsDim_, actDim_, obsDim_estim_, actDim_estim_;
  int historyLength_, historySamples_, historyStep_, iter_;
  float dt_;
  std::chrono::time_point<std::chrono::steady_clock> t_start_;
  std::chrono::time_point<std::chrono::steady_clock> t_end_;

  // History quantities
  const static int N_hist = 3;
  Eigen::Matrix<float, 3, N_hist> hist_base_ang_vel;
  Eigen::Matrix<float, 2, N_hist> hist_roll_pitch;
  Eigen::Matrix<float, 10, N_hist> hist_pos_lower, hist_vel_lower, hist_act_lower;

  // Related to leg phases
  Vector2 phases_freq_;
  Eigen::Array<float, 2, 1> phases_;

  // Related to orientation
  Vector3 _bodyOri, _bodyAngularVel, _gravityVec, _qa, _qb, _qc, _qvec;
  Vector4 _bodyQuat;

  bool load_metadata_scale_and_default_qref(const std::filesystem::path &model_path);
  std::vector<float> parse_csv_floats(const std::string &csv) const;
};

Interface::Interface() {

  // Fill history quantities
  hist_base_ang_vel.fill(0.0);
  hist_roll_pitch.fill(0.0);
  hist_pos_lower.fill(0.0);
  hist_vel_lower.fill(0.0);
  hist_act_lower.fill(0.0);

  has_meta_default_qref_ = false;

  // History is not used for now so we can hardcode 1s
  historySamples_ = 1;
  historyStep_ = 1;
  historyLength_ = 1 + historyStep_ * (historySamples_ - 1);
  iter_ = 0;
}

void Interface::initialize(std::basic_string<ORTCHAR_T> model_file,
                           const Vxf &q_ref, float dt) {

  // Initialize ONNX framework
  std::filesystem::path model_path(model_file);
  const bool has_extension = model_path.has_extension();
  std::filesystem::path base_path = model_path;
  if (has_extension) {
    base_path.replace_extension();
  }

  std::filesystem::path actor_path = model_path;
  if (!std::filesystem::exists(actor_path)) {
    std::cerr << "[Interface] Actor model not found. Checked: " << model_path
              << " and " << actor_path << std::endl;
  }

  policy_actor_ = std::make_shared<OnnxWrapper>(actor_path.native());
  policy_actor_->initialize();

  // policy_estim_ = std::make_shared<OnnxWrapper>(estim_file);
  // policy_estim_->initialize();

  // Retrieve info about networks
  obsDim_ = policy_actor_->get_obsDim();
  actDim_ = policy_actor_->get_actDim();
  std::cout << "Actor Network parameters: " << std::endl;
  std::cout << "obsDim: " << obsDim_ << " | actDim: " << actDim_ << std::endl;

  
  /*obsDim_estim_ = policy_estim_->get_obsDim();
  actDim_estim_ = policy_estim_->get_actDim();
  std::cout << "Actor Network parameters: " << std::endl;
  std::cout << "obsDim: " << obsDim_estim_ << " | actDim: " << actDim_estim_ << std::endl;*/

  // Initialize some tensors
  obs_ = Vxf::Zero(obsDim_);
  actorObs_ = Vxf::Zero(obsDim_);
  studentObs_ = Vxf::Zero(obsDim_ * historySamples_);
  historyObs_ = Vxf::Zero(obsDim_ * historyLength_);
  historyTempObs_ = Vxf::Zero(obsDim_ * historyLength_);
  actions_ = Vxf::Zero(actDim_);
  action_scale_ = Vxf::Ones(actDim_);
  last_actions_ = Vxf::Zero(actDim_);
  pTarget_ = Vxf::Zero(actDim_);
  estim_vel_ = Vxf::Zero(6);
  h_gru_ = Vxf::Zero(128);

  load_metadata_scale_and_default_qref(actor_path);
  /*last_actions_ = Eigen::MatrixXf::Zero(nJoints, 6);
  last_dof_pos_ = Eigen::MatrixXf::Zero(nJoints, 6);
  last_dof_vel_ = Eigen::MatrixXf::Zero(nJoints, 6);*/

  // Reference position around which to apply the actions
  if (has_meta_default_qref_ && meta_default_qref_.size() == q_ref.size()) {
    q_ref_ = meta_default_qref_;
  } else {
    q_ref_ = q_ref;
  }

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

/*
Vxf Interface::forward() {

  // Compute policy actions
  actions_ = policy_actor_->run(obs_);

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
*/

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

std::vector<float> Interface::parse_csv_floats(const std::string &csv) const {
  std::vector<float> out;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    try {
      out.push_back(std::stof(item));
    } catch (...) {
    }
  }
  return out;
}

bool Interface::load_metadata_scale_and_default_qref(const std::filesystem::path &model_path) {
  has_meta_default_qref_ = false;
  // Only available for actor model.
  if (!policy_actor_) {
    return false;
  }
  bool scale_set = false;
  std::string csv;
  if (policy_actor_->get_metadata_value("action_scale", csv)) {
    auto vals = parse_csv_floats(csv);
    if (!vals.empty()) {
      if (static_cast<int>(vals.size()) == actDim_) {
        action_scale_ = Eigen::Map<Vxf>(vals.data(), vals.size());
        scale_set = true;
      } else {
        std::cerr << "[Interface] action_scale metadata size mismatch: " << vals.size()
                  << " vs actDim " << actDim_ << std::endl;
      }
    }
  }
  if (policy_actor_->get_metadata_value("default_joint_pos", csv)) {
    auto vals = parse_csv_floats(csv);
    if (!vals.empty()) {
      if (static_cast<int>(vals.size()) == actDim_) {
        meta_default_qref_ = Eigen::Map<Vxf>(vals.data(), vals.size());
        has_meta_default_qref_ = true;
      } else {
        std::cerr << "[Interface] default_joint_pos metadata size mismatch: " << vals.size()
                  << " vs actDim " << actDim_ << std::endl;
      }
    }
  }
  // Fallback to known mjlab defaults if metadata not available (older ORT without metadata API).
  if (!scale_set && actDim_ == 19) {
    static const float kDefaultScale[19] = {0.2250f, 0.2250f, 0.2250f, 0.2330f, 0.2370f,
                                            0.2250f, 0.2250f, 0.2250f, 0.2330f, 0.2370f,
                                            0.1500f, 0.4380f, 0.4380f, 0.4380f, 0.4380f,
                                            0.4380f, 0.4380f, 0.4380f, 0.4380f};
    action_scale_ = Eigen::Map<const Vxf>(kDefaultScale, 19);
    std::cerr << "[Interface] action_scale metadata missing; using mjlab defaults." << std::endl;
  }
  if (!has_meta_default_qref_ && actDim_ == 19) {
    static const float kDefaultQref[19] = {0.0f,  -0.02f, -0.4f, 0.8f,  -0.4f,
                                           0.0f,  -0.02f, -0.4f, 0.8f,  -0.4f,
                                           0.0f,  0.0f,   0.0f,  0.0f,  0.0f,
                                           0.0f,  0.0f,   0.0f,  0.0f};
    meta_default_qref_ = Eigen::Map<const Vxf>(kDefaultQref, 19);
    has_meta_default_qref_ = true;
    std::cerr << "[Interface] default_joint_pos metadata missing; using mjlab defaults." << std::endl;
  }
  return has_meta_default_qref_;
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
  // mjlab ONNX uses motor/natural joint order already; keep identity.
  return v;
}

Vxf Interface::forward_ManiSkill() {

  // obs_ = Vxf::Ones(92);
  // h_gru_ = Vxf::Ones(128);

  // Compute velocity estimation
  // std::cout << "Build inputs " << std::endl;
  
  
  /*std::vector<Vxf> inputs_estimator = {obs_, h_gru_};
  std::vector<Vxf> outputs_estimator = policy_estim_->run(inputs_estimator);
  estim_vel_ = outputs_estimator.front();
  h_gru_ = outputs_estimator.back();*/
  
  //std::cout << "Estim " << estim_vel_.transpose() << std::endl;

  // Compute policy actions
  std::vector<Vxf> inputs_actor = {obs_}; //, estim_vel_};
  std::vector<Vxf> outputs_actor = policy_actor_->run(inputs_actor);
  actions_ = outputs_actor.front();

  //std::cout << "actions_  " << actions_ << std::endl;

  // Force torso to 0
  /*actions_[2] = 0.0;
  for (int i = 10; i < policy_actor_out_.rows(); ++i) {
    actions_[i] = 0.0;
  }*/

  // Force torso and arms to 0
  /*std::array<int, 9> idx = {2, 5, 6, 9, 10, 13, 14, 17, 18};
  for (size_t i = 0; i < idx.size(); ++i) {
    actions_[idx[i]] = 0.0;
  }
  actions_[2] = 0.0;*/

  // Force arms to 0 except pitch
  /* actions_[9] = 0.0;
  actions_[10] = 0.0;
  actions_[13] = 0.0;
  actions_[14] = 0.0;
  actions_[17] = 0.0;
  actions_[18] = 0.0; */

  /*if (iter_ < 200) {
  actions_ << -0.1506,  0.1241,  0.0025, -0.3543,  0.6767,  0.0049, -0.0076, -0.0899,
         -0.5155,  0.2288,  0.0218, -0.0599, -0.1345, -0.0472, -0.2006, -0.0569,
         -0.0259,  0.0666, -0.0727;}*/

  /*const Vector19 scale = {0.2250, 0.2250, 0.2250, 0.2333, 0.2375, 0.2250, 0.2250, 0.2250, 0.2333,
         0.2375, 0.1500, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375,
         0.4375};*/

  // actions_.fill(0.5);
    
  // Target joint positions based on scaled actions
  Vxf scale = action_scale_.size() == actions_.size() ? action_scale_ : Vxf::Ones(actions_.size());
  Vxf scaled_actions = scale.array() * actions_.array();
  if (scaled_actions.size() != q_ref_.rows()) {
    std::cerr << "[Interface] Action dimension mismatch: policy outputs "
              << scaled_actions.size() << ", expected " << q_ref_.rows() << std::endl;
    scaled_actions.conservativeResize(q_ref_.rows());
  }
  assert(q_ref_.rows() == scaled_actions.rows());
  pTarget_ = q_ref_ + scaled_actions; //  + 0.25 * last_actions_ ;

  std::cout << "== ACTIONS ==" << std::endl;
  std::cout << scale.array() << std::endl;
  std::cout << q_ref_.transpose() << std::endl;

  assert(pTarget_.rows() == q_ref_.rows());

  // last_actions_.col(0) = reorder_act(actions_);

  // Log time
  t_end_ = std::chrono::steady_clock::now();

  return pTarget_;
}

void Interface::update_observation_ManiSkill(
    const Vector19 &pos, const Vector19 &vel, const Vector19 &tau,
    const Vector3 &rpy, const Vector4 &ori, const Vector3 &gyro,
    const Vector6 &cmd, float time) {
  // Log time
  t_start_ = std::chrono::steady_clock::now();

  // Fast path for mjlab-style policy with obsDim ~= 66
  if (obsDim_ <= 70) {
    _bodyQuat = ori;
    transformBodyQuat();
    Vector3 projected_gravity = -1.0f * _bodyOri;

    // Flatten to: base_ang_vel(3), projected_gravity(3), pos-q_ref_(19), vel(19), actions_(19), cmd(x,y,yaw)
    Eigen::Index idx = 0;
    obs_.segment<3>(idx) = gyro;
    idx += 3;
    obs_.segment<3>(idx) = projected_gravity;
    idx += 3;
    obs_.segment<19>(idx) = pos - q_ref_;
    idx += 19;
    obs_.segment<19>(idx) = vel;
    idx += 19;
    obs_.segment<19>(idx) = actions_;
    idx += 19;
    obs_.segment<3>(idx) << cmd(0), cmd(1), cmd(5);
    idx += 3;
    if (idx != obsDim_) {
      std::cerr << "[Interface] Observation packing mismatch (compact path): expected "
                << obsDim_ << " got " << idx << std::endl;
    }
    iter_++;
    return;
  }

  // Projected gravity based on orientation state
  _bodyQuat = ori;
  transformBodyQuat(); // this update _bodyOri
  Vector3 projected_gravity = -1.0 * _bodyOri;

  //std::cout << projected_gravity.transpose() << std::endl;

  float roll = rpy(0);
  float pitch = rpy(1);

  float phase = 2 * pi_v * 1.2 * time;

  Vector3 base_ang_vel = gyro;
  // base_ang_vel.head(2) *= -1;  // Invert x y to abide by simulation convention
  
  Vector10 pos_lower, vel_lower, act_lower;
  Vector19 reordered_pos = pos; // reorder_obs(pos);
  Vector19 reordered_vel = vel; // reorder_obs(vel);

  std::array<int, 10> idx = {0, 1, 3, 4, 7, 8, 11, 12, 15, 16};
  for (size_t i = 0; i < idx.size(); ++i) {
    pos_lower[i] = reordered_pos[idx[i]];
    vel_lower[i] = reordered_vel[idx[i]];
    act_lower[i] = actions_[i];
  }

  // Roll history
  for (int i = hist_base_ang_vel.cols() - 1; i > 0; i--) {
    hist_base_ang_vel.col(i) = hist_base_ang_vel.col(i - 1);
    hist_roll_pitch.col(i) = hist_roll_pitch.col(i - 1);
    hist_pos_lower.col(i) = hist_pos_lower.col(i - 1);
    hist_vel_lower.col(i) = hist_vel_lower.col(i - 1);
    hist_act_lower.col(i) = hist_act_lower.col(i - 1);
  }
  hist_base_ang_vel.col(0) = base_ang_vel;
  hist_roll_pitch.col(0) = Vector2(roll, pitch);
  hist_pos_lower.col(0) = pos_lower;
  hist_vel_lower.col(0) = vel_lower;
  hist_act_lower.col(0) = act_lower;

  // TODO: Flatten with Map<VectorXf> v1(M1.data(), M1.size());
  // to reshape into a Vector with all columns flattened.

  //std::cout << hist_base_ang_vel.col(0).transpose() << std::endl;

  // Filling observation vector
  obs_ << gyro,
          projected_gravity,
          pos - q_ref_,
          vel,
          actions_,
          cmd(0),
          cmd(1),
          cmd(5);


  
          /*hist_base_ang_vel.col(0) * 0.25,
          hist_base_ang_vel.col(1) * 0.25,
          hist_base_ang_vel.col(2) * 0.25,
          hist_roll_pitch.col(0),
          hist_roll_pitch.col(1),
          hist_roll_pitch.col(2),
          hist_pos_lower.col(0),
          hist_pos_lower.col(1),
          hist_pos_lower.col(2),
          hist_vel_lower.col(0) * 0.05,
          hist_vel_lower.col(1) * 0.05,
          hist_vel_lower.col(2) * 0.05,
          hist_act_lower.col(0),
          hist_act_lower.col(1),
          hist_act_lower.col(2),*/
          /*reorder_obs(pos),
          reorder_obs(vel),
          actions_,
          loco_mode,*/
          /*std::cos(phase),
          std::sin(phase),
          cmd(0),
          cmd(1),
          cmd(5);*/
          /*Vxf::Zero(6),
          hist_base_ang_vel.col(0),
          hist_roll_pitch.col(0),
          hist_pos_lower.col(0),
          hist_vel_lower.col(0),
          hist_act_lower.col(0),
          std::cos(phase),
          std::sin(phase),
          cmd(0),
          cmd(1),
          cmd(5),
          Vxf::Zero(6);*/
          // Vxf::Zero(4); // Unused by actor but was there for critic
          //Vxf::Zero(319); // Unused by actor

  // std::cout << "= STEP = " << std::endl;
  // std::cout << obs_.transpose() << std::endl;
  // if (time > 0.1) {exit(-1);}

  // obs_.fill(1.0);

  // std::cout << obs_.rows() << obsDim_ << std::endl;

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

  // obs_ = obs_ * 0.0 + Vxf::Ones(obsDim_);

  // Iteration counter
  iter_++;
}