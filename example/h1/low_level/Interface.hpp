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
  /// parameters
  /// \param[in] dt Control time step
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(std::basic_string<ORTCHAR_T> model_file, float dt);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Forward pass
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vxf forward();

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
  Vxf get_default_qref();

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
                           float dt) {

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
  if (has_meta_default_qref_ && meta_default_qref_.size() == 19) {
    q_ref_ = meta_default_qref_;
  } else {
    throw std::runtime_error("Default qref not found.");
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
    static const float kDefaultQref[19] = {0.0f,  -0.0f, -0.2f, 0.6f,  -0.4f,
                                           0.0f,  -0.0f, -0.2f, 0.6f,  -0.4f,
                                           0.0f,  0.0f,   0.0f,  0.0f,  0.0f,
                                           0.0f,  0.0f,   0.0f,  0.0f};
    meta_default_qref_ = Eigen::Map<const Vxf>(kDefaultQref, 19);
    has_meta_default_qref_ = true;
    std::cerr << "[Interface] default_joint_pos metadata missing; using mjlab defaults." << std::endl;
  }
  return has_meta_default_qref_;
}

Vxf Interface::get_default_qref() {
  assert(has_meta_default_qref_);
  return meta_default_qref_;
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
  historyObs_.head(obsDim_) = obs_;

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

Vxf Interface::forward() {

  // Compute policy actions
  std::vector<Vxf> inputs_actor = {obs_};
  std::vector<Vxf> outputs_actor = policy_actor_->run(inputs_actor);
  actions_ = outputs_actor.front();
    
  // Target joint positions based on scaled actions
  Vxf scale = action_scale_.size() == actions_.size() ? action_scale_ : Vxf::Ones(actions_.size());
  Vxf scaled_actions = scale.array() * actions_.array();
  if (scaled_actions.size() != q_ref_.rows()) {
    std::cerr << "[Interface] Action dimension mismatch: policy outputs "
              << scaled_actions.size() << ", expected " << q_ref_.rows() << std::endl;
    scaled_actions.conservativeResize(q_ref_.rows());
  }
  assert(q_ref_.rows() == scaled_actions.rows());
  pTarget_ = q_ref_ + scaled_actions;

  assert(pTarget_.rows() == q_ref_.rows());

  // last_actions_.col(0) = reorder_act(actions_);

  // Log time
  t_end_ = std::chrono::steady_clock::now();

  return pTarget_;
}

void Interface::update_observation(
    const Vector19 &pos, const Vector19 &vel, const Vector19 &tau,
    const Vector3 &rpy, const Vector4 &ori, const Vector3 &gyro,
    const Vector6 &cmd, float time) {
  // Log time
  t_start_ = std::chrono::steady_clock::now();

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