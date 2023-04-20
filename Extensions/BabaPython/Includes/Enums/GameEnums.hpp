// Copyright (c) 2020-2023 Chris Ohk

// I am making my contributions/submissions to this project solely in our
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef BABA_IS_AUTO_PYTHON_GAME_ENUMS_HPP
#define BABA_IS_AUTO_PYTHON_GAME_ENUMS_HPP

#include <pybind11/pybind11.h>

void AddGameEnums(pybind11::module& m);
void AddGameEnumUtils(pybind11::module& m);

#endif  // BABA_IS_AUTO_PYTHON_GAME_ENUMS_HPP