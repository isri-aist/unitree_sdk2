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
  OnnxWrapper(){};

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
  ~OnnxWrapper(){};

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
  /// \param[in] vs List of vectors of observations
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<Vxf> run(const std::vector<Vxf> &vs);

  int get_obsDim(int i = 0) { return total_numbers_elements_.at(i); }
  int get_actDim(int i = 0) { return calculate_product(outputs_shapes_.at(0)); }

private:
  std::shared_ptr<Ort::Session> session_; // ONNX Runtime session

  std::vector<std::vector<std::int64_t>> inputs_shapes_;  // Shapes of the inputs of the network
  std::vector<std::vector<std::int64_t>> outputs_shapes_;  // Shapes of the outputs of the network
  std::vector<std::string> inputs_names_;  // Names of the inputs of the network
  std::vector<std::string> outputs_names_;  // Names of the outputs of the network

  std::vector<int> total_numbers_elements_; // Total number of elements in the inputs of the
                                            // network
};

OnnxWrapper::OnnxWrapper(std::basic_string<ORTCHAR_T> model_file) {
  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
  Ort::SessionOptions session_options;
  session_ =
      std::make_shared<Ort::Session>(env, model_file.c_str(), session_options);
}

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

  std::cout << "== Initialize ONNX Wrapper ==" << std::endl;
  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::cout << "Input Node Name/Shape (" << inputs_names_.size()
            << "):" << std::endl;
  for (std::size_t i = 0; i < session_->GetInputCount(); i++) {
    inputs_names_.emplace_back(
        session_->GetInputNameAllocated(i, allocator).get());
    inputs_shapes_.push_back(
        session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cout << "\t" << inputs_names_.at(i) << " : "
              << print_shape(inputs_shapes_.at(i)) << std::endl;

    // some models might have negative shape values to indicate dynamic shape,
    // e.g., for variable batch size.
    for (auto &s : inputs_shapes_.at(i)) {
      if (s < 0) {
        s = 1;
      }
    }
  }

  // print name/shape of outputs
  std::cout << "Output Node Name/Shape (" << outputs_names_.size()
            << "):" << std::endl;
  for (std::size_t i = 0; i < session_->GetOutputCount(); i++) {
    outputs_names_.emplace_back(
        session_->GetOutputNameAllocated(i, allocator).get());
    outputs_shapes_.push_back(
        session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cout << "\t" << outputs_names_.at(i) << " : "
              << print_shape(outputs_shapes_.at(i)) << std::endl;
  }

  // Assume model has 2 input nodes at max and 2 output nodes.
  assert(inputs_names_.size() <= 2 && outputs_names_.size() <= 2);

  // Compute total sizes of inputs
  for (std::size_t i = 0; i < inputs_names_.size(); i++) {
    total_numbers_elements_.push_back(calculate_product(inputs_shapes_.at(i)));
  }

  // Create Ort tensors of random numbers in the range [0, 255]
  std::cout << "Constructing dummy tensors of random values." << std::endl;
  std::vector<Ort::Value> inputs_tensors;
  for (std::size_t i = 0; i < inputs_names_.size(); i++) {
    std::vector<float> input_tensor_values(total_numbers_elements_.at(i));
    std::generate(input_tensor_values.begin(), input_tensor_values.end(),
                [&] { return rand() % 255; });

    inputs_tensors.emplace_back(
        vec_to_tensor<float>(input_tensor_values, inputs_shapes_.at(i)));

    // Double-check the dimensions of the input tensor
    assert(inputs_tensors.at(i).IsTensor() &&
           inputs_tensors.at(i).GetTensorTypeAndShapeInfo().GetShape() == inputs_shapes_.at(i));
  
    std::cout << "input_tensor shape: "
              << print_shape(
                    inputs_tensors.at(i).GetTensorTypeAndShapeInfo().GetShape())
              << std::endl;
  }

  // Convert names to C strings
  std::vector<const char *> inputs_names_char(inputs_names_.size(), nullptr);
  std::transform(std::begin(inputs_names_), std::end(inputs_names_),
                 std::begin(inputs_names_char),
                 [&](const std::string &str) { return str.c_str(); });

  std::vector<const char *> outputs_names_char(outputs_names_.size(), nullptr);
  std::transform(std::begin(outputs_names_), std::end(outputs_names_),
                 std::begin(outputs_names_char),
                 [&](const std::string &str) { return str.c_str(); });

  // Pass random data through model
  std::cout << "Running model once to check validity..." << std::endl;
  try {
    auto outputs_tensors =
        session_->Run(Ort::RunOptions{nullptr}, inputs_names_char.data(),
                      inputs_tensors.data(), inputs_names_char.size(),
                      outputs_names_char.data(), outputs_names_char.size());
    std::cout << "Done!" << std::endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes
    // specifed in the Run() call
    assert(outputs_tensors.size() == outputs_names_.size() &&
           outputs_tensors.at(0).IsTensor());

    for (std::size_t i = 0; i < outputs_tensors.size(); i++) {
      std::cout << "output_tensor shape: "
                << print_shape(
                      outputs_tensors.at(i).GetTensorTypeAndShapeInfo().GetShape())
                << std::endl;
    }


  } catch (const Ort::Exception &exception) {
    std::cout << "ERROR running model inference: " << exception.what()
              << std::endl;
    exit(-1);
  }
}

std::vector<Vxf> OnnxWrapper::run(const std::vector<Vxf> &vs) {

  // std::cout << "VS Content" << std::endl;
  // std::cout << vs.front().transpose() << std::endl;
  // std::cout << vs.back().transpose() << std::endl;

  // Check size of the input vectors
  assert(vs.size() == inputs_names_.size());

  for (std::size_t i = 0; i < inputs_names_.size(); i++) {
    assert(vs.at(i).size() == total_numbers_elements_.at(i));
  }

  // Convert Eigen vector to std vector, then to ONNX Runtime tensors
  std::vector<std::vector<float>> inputs_std_v;
  for (std::size_t i = 0; i < inputs_names_.size(); i++) {
    std::vector<float> std_v;
    std_v.resize(vs.at(i).size());
    inputs_std_v.push_back(std_v);
  }

  std::vector<Ort::Value> inputs_tensors;

  for (std::size_t i = 0; i < inputs_names_.size(); i++) {
    //std::vector<float> std_v;
    //std_v.resize(vs.at(i).size());
    //Vxf::Map(&std_v[0], vs.at(i).size()) = vs.at(i);

    Vxf::Map(&(inputs_std_v.at(i))[0], vs.at(i).size()) = vs.at(i);

    /*Ort::Value tensor = vec_to_tensor<float>(std_v, inputs_shapes_.at(i));
    float *ftensor = tensor.GetTensorMutableData<float>();
    std::cout << "ftensor" << std::endl;
    for (int i = 0; i < 92; i++) {
      std::cout << i << " " << ftensor[i] << std::endl;
    }*/

    inputs_tensors.emplace_back(vec_to_tensor<float>(inputs_std_v.at(i), inputs_shapes_.at(i)));

    /*float *floatarrinq = inputs_tensors.front().GetTensorMutableData<float>();
    std::cout << "inside loop" << std::endl;
    for (int i = 0; i < 15; i++) {
      std::cout << i << " " << floatarrinq[i] << std::endl;
    }*/
  }

  /*float *floatarrinq = inputs_tensors.front().GetTensorMutableData<float>();
  std::cout << "ini tensor" << std::endl;
  for (int i = 0; i < 92; i++) {
    std::cout << i << " " << floatarrinq[i] << std::endl;
  }*/

  // Double-check the dimensions of the inputs tensors
  for (std::size_t i = 0; i < inputs_shapes_.size(); i++) {
    assert(inputs_tensors.at(i).IsTensor() &&
           inputs_tensors.at(i).GetTensorTypeAndShapeInfo().GetShape() == inputs_shapes_.at(i));
  }

  // Convert names to C strings
  std::vector<const char *> inputs_names_char(inputs_names_.size(), nullptr);
  std::transform(std::begin(inputs_names_), std::end(inputs_names_),
                 std::begin(inputs_names_char),
                 [&](const std::string &str) { return str.c_str(); });

  std::vector<const char *> outputs_names_char(outputs_names_.size(), nullptr);
  std::transform(std::begin(outputs_names_), std::end(outputs_names_),
                 std::begin(outputs_names_char),
                 [&](const std::string &str) { return str.c_str(); });

  // Get pointer to output tensor float values
  /*float *floatarrin = inputs_tensors.front().GetTensorMutableData<float>();
  std::cout << "in tensor" << std::endl;
  for (int i = 0; i < 92; i++) {
    std::cout << i << " " << floatarrin[i] << std::endl;
  }
  float *floatarrina = inputs_tensors.back().GetTensorMutableData<float>();
  std::cout << "in tensor" << std::endl;
  for (int i = 0; i < 128; i++) {
    std::cout << i << " " << floatarrina[i] << std::endl;
  }*/

  // Run inference
  try {
    auto output_tensors =
        session_->Run(Ort::RunOptions{nullptr}, inputs_names_char.data(),
                      inputs_tensors.data(), inputs_names_char.size(),
                      outputs_names_char.data(), outputs_names_char.size());
    // Double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes
    // specifed in the Run() call
    assert(output_tensors.size() == outputs_names_.size() &&
           output_tensors.at(0).IsTensor());

    // Create output vector that will contain the results
    std::vector<Vxf> outputs;
    for (std::size_t i = 0; i < output_tensors.size(); i++) {
      std::vector<std::int64_t> tensor_shape = output_tensors.at(i).GetTensorTypeAndShapeInfo().GetShape();
      int tensor_size = calculate_product(tensor_shape);
      assert(tensor_size == calculate_product(outputs_shapes_.at(i)));
      Vxf output = Vxf::Zero(tensor_size);
      
      // Get pointer to output tensor float values
      float *floatarr = output_tensors.at(i).GetTensorMutableData<float>();
      for (int i = 0; i < output.size(); i++) {
        output[i] = floatarr[i];
      }

      outputs.push_back(output);
      // std::cout << "OUT" << outputs.back().transpose() << std::endl;
    }

    return outputs;

  } catch (const Ort::Exception &exception) {
    std::cout << "ERROR running model inference: " << exception.what()
              << std::endl;
    exit(-1);
  }
}

#endif // ONNXWRAPPER_H_INCLUDED