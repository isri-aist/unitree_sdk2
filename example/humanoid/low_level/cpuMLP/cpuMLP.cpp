#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(interface, m) {
  py::class_<Interface>(m, "Interface")
      .def(py::init<int, int, int, int, int>())
      .def("initialize", &Interface::initialize)
      .def("forward", &Interface::forward)
      .def("update_observation", &Interface::update_observation)
      .def("get_observation", &Interface::get_observation)
      .def("get_computation_time", &Interface::get_computation_time)
      .def_readwrite("pTarget12", &Interface::pTarget12_)
      .def_readwrite("q_init_", &Interface::q_init_)
      .def_readwrite("vel_command", &Interface::vel_command_)
      .def_readwrite("obs_", &Interface::obs_)
      .def_readwrite("actorObs_", &Interface::actorObs_)
      .def_readwrite("historyObs_", &Interface::historyObs_)
      .def_readwrite("bound_pi_", &Interface::bound_pi_);
}
