///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for OnnxWrapper class
///
/// \details This class handles interaction with ONNX Runtime to run inferences.
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ONNXWRAPPER_H_INCLUDED
#define ONNXWRAPPER_INCLUDED

#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <sstream>
#include <string>
#include <vector>

#include "Types.h"

class OnnxWrapper {
public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  OnnxWrapper() : session_(nullptr){};

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  /// \param[in] model_file Path to the .onnx model file
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  OnnxWrapper(std::basic_string<ORTCHAR_T> model_file);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~OnnxWrapper();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Pretty print of a shape dimension vector
  ///
  /// \param[in] v The vector storing the dimensions to print
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  std::string print_shape(const std::vector<std::int64_t> &v);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the total number of elements in the input of the network
  ///
  /// \param[in] v The vector storing the shape of the input
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int calculate_product(const std::vector<std::int64_t> &v);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Convert a vector into a corresponding tensor of the desired shape
  ///
  /// \param[in] v Vector to be converted
  /// \param[in] shape Desired shape of the output tensor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  Ort::Value vec_to_tensor(std::vector<T> &data,
                           const std::vector<std::int64_t> &shape);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize the network
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run network inference to get policy actions
  ///
  /// \param[in] v Vector of observations
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  VectorM run(Eigen::VectorXf v);

private:
  Ort::Session *session_; // ONNX Runtime session

  std::vector<std::int64_t> input_shapes_; // Shape of the inputs of the network
  std::vector<std::int64_t>
      output_shapes_; // Shape of the outputs of the network

  std::vector<std::string> input_names_;  // Name of the inputs of the network
  std::vector<std::string> output_names_; // Name of the outputs of the network

  std::vector<Ort::Value> input_tensors_; // Input tensors of the network

  int total_number_elements_; // Total number of elements in the input of the
                              // network
};

OnnxWrapper::OnnxWrapper(std::basic_string<ORTCHAR_T> model_file) {
  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
  Ort::SessionOptions session_options;
  session_ = new Ort::Session(env, model_file.c_str(), session_options);
}

OnnxWrapper::~OnnxWrapper() {}

std::string OnnxWrapper::print_shape(const std::vector<std::int64_t> &v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int OnnxWrapper::calculate_product(const std::vector<std::int64_t> &v) {
  int total = 1;
  for (auto &i : v)
    total *= i;
  return total;
}

template <typename T>
Ort::Value OnnxWrapper::vec_to_tensor(std::vector<T> &data,
                                      const std::vector<std::int64_t> &shape) {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(),
                                            shape.data(), shape.size());
  return tensor;
}

void OnnxWrapper::initialize() {

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::cout << "Input Node Name/Shape (" << input_names_.size()
            << "):" << std::endl;
  for (std::size_t i = 0; i < session_->GetInputCount(); i++) {
    input_names_.emplace_back(
        session_->GetInputNameAllocated(i, allocator).get());
    input_shapes_ =
        session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << input_names_.at(i) << " : "
              << print_shape(input_shapes_) << std::endl;
  }
  // some models might have negative shape values to indicate dynamic shape,
  // e.g., for variable batch size.
  for (auto &s : input_shapes_) {
    if (s < 0) {
      s = 1;
    }
  }

  // print name/shape of outputs
  std::cout << "Output Node Name/Shape (" << output_names_.size()
            << "):" << std::endl;
  for (std::size_t i = 0; i < session_->GetOutputCount(); i++) {
    output_names_.emplace_back(
        session_->GetOutputNameAllocated(i, allocator).get());
    output_shapes_ =
        session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << output_names_.at(i) << " : "
              << print_shape(output_shapes_) << std::endl;
  }

  // Assume model has 1 input node and 1 output node.
  assert(input_names_.size() == 1 && output_names_.size() == 1);

  // Create a single Ort tensor of random numbers
  auto input_shape = input_shapes_;
  total_number_elements_ = calculate_product(input_shape);

  // generate random numbers in the range [0, 255]
  std::vector<float> input_tensor_values(total_number_elements_);
  std::generate(input_tensor_values.begin(), input_tensor_values.end(),
                [&] { return rand() % 255; });
  input_tensors_.emplace_back(
      vec_to_tensor<float>(input_tensor_values, input_shape));

  // double-check the dimensions of the input tensor
  assert(input_tensors_[0].IsTensor() &&
         input_tensors_[0].GetTensorTypeAndShapeInfo().GetShape() ==
             input_shape);
  std::cout << "\ninput_tensor shape: "
            << print_shape(
                   input_tensors_[0].GetTensorTypeAndShapeInfo().GetShape())
            << std::endl;

  // pass data through model
  std::vector<const char *> input_names_char(input_names_.size(), nullptr);
  std::transform(std::begin(input_names_), std::end(input_names_),
                 std::begin(input_names_char),
                 [&](const std::string &str) { return str.c_str(); });

  std::vector<const char *> output_names_char(output_names_.size(), nullptr);
  std::transform(std::begin(output_names_), std::end(output_names_),
                 std::begin(output_names_char),
                 [&](const std::string &str) { return str.c_str(); });

  std::cout << "Running model once to check validity..." << std::endl;

  try {
    auto output_tensors =
        session_->Run(Ort::RunOptions{nullptr}, input_names_char.data(),
                      input_tensors_.data(), input_names_char.size(),
                      output_names_char.data(), output_names_char.size());
    std::cout << "Done!" << std::endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes
    // specifed in the Run() call
    assert(output_tensors.size() == output_names_.size() &&
           output_tensors[0].IsTensor());
  } catch (const Ort::Exception &exception) {
    std::cout << "ERROR running model inference: " << exception.what()
              << std::endl;
    exit(-1);
  }
}

VectorM OnnxWrapper::run(Eigen::VectorXf v) {

  // Check size of the input vector
  assert(v.size() == total_number_elements_);

  // Convert Eigen vector to std vector
  std::vector<float> std_v;
  std_v.resize(v.size());
  Eigen::VectorXf::Map(&std_v[0], v.size()) = v;

  // Convert std vector to ONNX Runtime tensor
  input_tensors_.emplace_back(vec_to_tensor<float>(std_v, input_shapes_));

  // Double-check the dimensions of the input tensor
  assert(input_tensors_[0].IsTensor() &&
         input_tensors_[0].GetTensorTypeAndShapeInfo().GetShape() ==
             input_shapes_);

  // pass data through model
  std::vector<const char *> input_names_char(input_names_.size(), nullptr);
  std::transform(std::begin(input_names_), std::end(input_names_),
                 std::begin(input_names_char),
                 [&](const std::string &str) { return str.c_str(); });

  std::vector<const char *> output_names_char(output_names_.size(), nullptr);
  std::transform(std::begin(output_names_), std::end(output_names_),
                 std::begin(output_names_char),
                 [&](const std::string &str) { return str.c_str(); });

  // Run inference
  try {
    auto output_tensors =
        session_->Run(Ort::RunOptions{nullptr}, input_names_char.data(),
                      input_tensors_.data(), input_names_char.size(),
                      output_names_char.data(), output_names_char.size());
    // Double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes
    // specifed in the Run() call
    assert(output_tensors.size() == output_names_.size() &&
           output_tensors[0].IsTensor());

    VectorM output = VectorM::Zero();
    assert(1 == output_shapes_[0] && output.size() == output_shapes_[1]);

    // Get pointer to output tensor float values
    float *floatarr = output_tensors.front().GetTensorMutableData<float>();
    for (int i = 0; i < output.size(); i++) {
      output[i] = floatarr[i];
    }

    return output;

  } catch (const Ort::Exception &exception) {
    std::cout << "ERROR running model inference: " << exception.what()
              << std::endl;
    exit(-1);
  }
}

#endif // ONNXWRAPPER_H_INCLUDED