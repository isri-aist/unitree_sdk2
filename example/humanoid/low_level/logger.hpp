#pragma once

#define FMT_HEADER_ONLY
#include "fmtlog-inl.h"
#include "fmtlog.h"
#include "base_state.h"
#include "motors.hpp"
#include <array>

template <typename T, size_t N>
std::string arrayToStringView(const std::array<T, N> &arr) {
  std::ostringstream oss;
  for (size_t i = 0; i < N; ++i) {
    oss << arr[i];
    if (i < N - 1) {
      oss << ",";
    }
  }
  return oss.str();
}

template <> struct fmt::formatter<BaseState> : formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(BaseState bs, FormatContext &ctx) {

    const std::string quat = "quat," + arrayToStringView(bs.quat);
    const std::string rpy = "rpy," + arrayToStringView(bs.rpy);
    const std::string omega = "omega," + arrayToStringView(bs.omega);
    const std::string acc = "acc," + arrayToStringView(bs.acc);
    const std::string all = quat + "\n" + rpy + "\n" + omega + "\n" + acc;

    string_view name = all;

    return formatter<string_view>::format(name, ctx);
  }
};

template <> struct fmt::formatter<MotorState> : formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(MotorState ms, FormatContext &ctx) {

    const std::string q = "q," + arrayToStringView(ms.q);
    const std::string dq = "dq," + arrayToStringView(ms.dq);
    const std::string tau = "tau," + arrayToStringView(ms.tau);
    const std::string all = q + "\n" + dq + "\n" + tau;

    string_view name = all;

    return formatter<string_view>::format(name, ctx);
  }
};

template <> struct fmt::formatter<MotorCommand> : formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(MotorCommand mc, FormatContext &ctx) {

    const std::string q_ref = "q_ref," + arrayToStringView(mc.q_ref);
    const std::string dq_ref = "dq_ref," + arrayToStringView(mc.dq_ref);
    const std::string kp = "kp," + arrayToStringView(mc.kp);
    const std::string kd = "kd," + arrayToStringView(mc.kd);
    const std::string tau_ff = "tau_ff," + arrayToStringView(mc.tau_ff);
    const std::string all =
        q_ref + "\n" + dq_ref + "\n" + kp + "\n" + kd + "\n" + tau_ff;

    string_view name = all;

    return formatter<string_view>::format(name, ctx);
  }
};