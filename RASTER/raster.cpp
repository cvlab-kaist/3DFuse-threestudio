/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <torch/extension.h>

#include "rasterize_points/rasterize_points.h"
#include "rasterize_coarse/rasterize_coarse.h"
#include "compositing/alpha_composite.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_points", &RasterizePoints);
  m.def("_rasterize_points_coarse", &RasterizePointsCoarse);
  m.def("_rasterize_points_naive", &RasterizePointsNaive);
  m.def("accum_alphacomposite", &alphaCompositeForward);
}
