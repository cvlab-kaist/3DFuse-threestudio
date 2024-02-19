# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import raster

# Example functions for blending the top K features per pixel using the outputs
# from rasterization.
# NOTE: All blending function should return a (N, H, W, C) tensor per batch element.
# This can be an image (C=3) or a set of features.


class _CompositeAlphaPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using alpha compositing. Given a z-buffer
    with corresponding features and weights, these values are accumulated according
    to their weights such that features nearer in depth contribute more to the final
    feature than ones further away.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
        cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    def forward(ctx, features, alphas, points_idx):
        pt_cld = raster.accum_alphacomposite(features, alphas, points_idx)

        # ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld


def alpha_composite(pointsidx, alphas, pt_clds) -> torch.Tensor:
    """
    Composite features within a z-buffer using alpha compositing. Given a z-buffer
    with corresponding features and weights, these values are accumulated according
    to their weights such that features nearer in depth contribute more to the final
    feature than ones further away.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
        cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])


    Args:
        pt_clds: Tensor of shape (N, C, P) giving the features of each point (can use
            RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[n, :, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    return _CompositeAlphaPoints.apply(pt_clds, alphas, pointsidx)

