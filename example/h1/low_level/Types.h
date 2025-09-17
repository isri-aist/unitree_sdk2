#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

// DYNAMIC SIZED TYPES
using Vxf = Eigen::VectorXf;
using Vxi = Eigen::VectorXi;
using Mxf = Eigen::MatrixXf;
using Matrix6N = Eigen::Matrix<float, 6, Eigen::Dynamic>;

// FIXED SIZED TYPES
using Vector1 = Eigen::Matrix<float, 1, 1>;
using Vector2 = Eigen::Matrix<float, 2, 1>;
using Vector3 = Eigen::Matrix<float, 3, 1>;
using Vector4 = Eigen::Matrix<float, 4, 1>;
using Vector5 = Eigen::Matrix<float, 5, 1>;
using Vector6 = Eigen::Matrix<float, 6, 1>;
using Vector7 = Eigen::Matrix<float, 7, 1>;
using Vector8 = Eigen::Matrix<float, 8, 1>;
using Vector9 = Eigen::Matrix<float, 9, 1>;
using Vector10 = Eigen::Matrix<float, 10, 1>;
using Vector19 = Eigen::Matrix<float, 19, 1>;
using Vector20 = Eigen::Matrix<float, 20, 1>;

using VectorM = Vector10;

#endif  // TYPES_H_INCLUDED
