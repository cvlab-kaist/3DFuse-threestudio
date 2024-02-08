#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class PointsRasterizationSettings:
    """
    Class to store the point rasterization params with defaults

    Members:
        image_size: Either common height and width or (height, width), in pixels.
        radius: The radius (in NDC units) of each disk to be rasterized.
            This can either be a float in which case the same radius is used
            for each point, or a torch.Tensor of shape (N, P) giving a radius
            per point in the batch.
        points_per_pixel: (int) Number of points to keep track of per pixel.
            We return the nearest points_per_pixel points along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts
            to set it heuristically based on the shape of the input. This should
            not affect the output, but can affect the speed of the forward pass.
        max_points_per_bin: Only applicable when using coarse-to-fine
            rasterization (bin_size != 0); this is the maximum number of points
            allowed within each bin. This should not affect the output values,
            but can affect the memory usage in the forward pass.
            Setting max_points_per_bin=None attempts to set with a heuristic.
    """

    image_size: Union[int, Tuple[int, int]] = 256
    radius: Union[float, torch.Tensor] = 0.01
    points_per_pixel: int = 8
    bin_size: Optional[int] = None
    max_points_per_bin: Optional[int] = None
