#include <eigen3/Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Types.h"
#include "iostream"

////// General MLP class with Dynamic hidden sizes
//
class MLP {
 public:
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Action;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1> State;

  MLP() {
    // Empty constructor
  }

  MLP(std::vector<int> hiddensizes, int StateDim, int ActionDim) {

    layersizes.push_back(StateDim);
    layersizes.reserve(layersizes.size() + hiddensizes.size());
    layersizes.insert(layersizes.end(), hiddensizes.begin(), hiddensizes.end());
    layersizes.push_back(ActionDim);
    ///[input hidden output]

    params.resize(2 * (layersizes.size() - 1));
    Ws.resize(layersizes.size() - 1);
    bs.resize(layersizes.size() - 1);
    lo.resize(layersizes.size());

    for (int i = 0; i < (int)(layersizes.size()); i++) {
      lo[i].resize(layersizes[i], 1);
      lo[i].setZero();
    }

    for (int i = 0; i < (int)(params.size()); i++) {
      //      int paramSize = 0;

      if (i % 2 == 0)  /// W resize
      {
        Ws[i / 2].resize(layersizes[i / 2 + 1], layersizes[i / 2]);
        params[i].resize(layersizes[i / 2] * layersizes[i / 2 + 1]);
        Ws[i / 2].setZero();
        params[i].setZero();
      }
      if (i % 2 == 1)  /// b resize
      {
        bs[(i - 1) / 2].resize(layersizes[(i + 1) / 2]);
        bs[(i - 1) / 2].setZero();
        params[i].resize(layersizes[(i + 1) / 2]);
        params[i].setZero();
      }
    }
  }

  void updateParamFromTxt(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::cout << "Loading " + fileName << std::endl;
    std::ifstream indata;
    indata.open(fileName);
    if (!indata.good()) {
      throw std::runtime_error("-- Failed to open --");
    }

    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;

    int totalN = 0;
    /// assign parameters
    for (int i = 0; i < (int)(params.size()); i++) {
      int paramSize = 0;
      while (std::getline(lineStream, cell, ',')) {  /// Read param
        params[i](paramSize++) = std::stof(cell);
        if (paramSize == params[i].size()) {
          break;
        }
      }
      totalN += paramSize;
      if (i % 2 == 0)  /// W copy
        memcpy(Ws[i / 2].data(), params[i].data(), sizeof(float) * Ws[i / 2].size());
      if (i % 2 == 1)  /// b copy
        memcpy(bs[(i - 1) / 2].data(), params[i].data(), sizeof(float) * bs[(i - 1) / 2].size());
    }
  }

  inline Action forward(State &state) {
    // state = (state - running_mean).cwiseProduct(div_running_var_sqrt);
    // std::cout << "State in " << state << std::endl << "State normed: " << state.cwiseMin(5.0).cwiseMax(-5.0) <<
    // std::endl;
    lo[0] = state.cwiseMin(5.0).cwiseMax(-5.0);  // Clamp in [-5, 5] range
    for (int cnt = 0; cnt < (int)(Ws.size()) - 1; cnt++) {
      lo[cnt + 1] = Ws[cnt] * lo[cnt] + bs[cnt];
      // LEAKYRELU activation
      // lo[cnt + 1] = lo[cnt + 1].cwiseMax(1e-2*lo[cnt + 1]);
      //
      // ELU activation
      // lo[cnt + 1] = lo[cnt + 1].cwiseMax(0.0) + (lo[cnt + 1].array().exp() - 1.0).matrix().cwiseProduct((lo[cnt +
      // 1].array() < 0.0).cast<double>().matrix());
      for (int i = 0; i < lo[cnt + 1].size(); i++)
        lo[cnt + 1][i] = lo[cnt + 1][i] > 0. ? lo[cnt + 1][i] : 1.0 * (std::exp(lo[cnt + 1][i]) - 1);
    }

    lo[lo.size() - 1] = Ws[Ws.size() - 1] * lo[lo.size() - 2] + bs[bs.size() - 1];  /// output layer
    if ((lo.back().array() != lo.back().array()).all()) std::cout << "state 2 nan" << std::endl;
    return lo.back();
  }

 private:
  std::vector<Eigen::Matrix<float, -1, 1>> params;
  std::vector<Eigen::Matrix<float, -1, -1>> Ws;
  std::vector<Eigen::Matrix<float, -1, 1>> bs;
  std::vector<Eigen::Matrix<float, -1, 1>> lo;

  std::vector<int> layersizes;
  bool isTanh = false;

};

////// General Scaler class to scale observations
//
class Scaler {
 public:
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1> State;

  Scaler() {
    // Empty constructor
  }

  Scaler(int StateDim) {
    running_mean = State::Zero(StateDim);
    running_var = State::Ones(StateDim);
    div_running_var_sqrt = State::Ones(StateDim);
  }

  void updateRunningMeanFromTxt(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::cout << "Loading " + fileName << std::endl;
    std::ifstream indata;
    indata.open(fileName);
    if (!indata.good()) {
      throw std::runtime_error("-- Failed to open --");
    }

    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;

    int meanSize = 0;
    while (std::getline(lineStream, cell, ',') && meanSize < running_mean.size()) {
      running_mean(meanSize++) = std::stof(cell);
    }

    std::cout << "Running mean: " << running_mean.transpose() << std::endl;
  }

  void updateRunningVarFromTxt(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::cout << "Loading " + fileName << std::endl;
    std::ifstream indata;
    indata.open(fileName);
    if (!indata.good()) {
      throw std::runtime_error("-- Failed to open --");
    }

    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;

    int varSize = 0;
    while (std::getline(lineStream, cell, ',') && varSize < running_var.size()) {
      running_var(varSize++) = std::stof(cell);
    }

    // Ensure no division by 0
    running_var = running_var.cwiseMax(1e-8);

    // Avoid dividing and computing square root for every call to forward()
    div_running_var_sqrt = running_var.cwiseSqrt().cwiseInverse();

    std::cout << "Running var: " << running_var.transpose() << std::endl;
  }

  inline State scale(State &state) {
    return (state - running_mean).cwiseProduct(div_running_var_sqrt);
  }

 private:
    State running_mean, running_var, div_running_var_sqrt;
};
