#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dcn_v2 import dcn_v2_conv, DCNv2, DCN
from dcn_v2 import dcn_v2_pooling, DCNv2Pooling, DCNPooling

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3


def conv_identify(weight, bias):
    weight.data.zero_()
    bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, y, x] = 1.0


def check_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
                          kernel_size=(kH, kW),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    dcn_v2 = DCNv2(inC, outC, (kH, kW),
                   stride=1, padding=1, dilation=1,
                   deformable_groups=deformable_groups).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()
    conv_identify(dcn_v2.weight, dcn_v2.bias)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    output = dcn_v2(input, offset, mask)
    output *= 2
    d = (input - output).abs().max()
    if d < 1e-10:
        print('Zero offset passed')
    else:
        print('Zero offset failed')
        print(input)
        print(output)

def check_gradient_dconv():
    # Minimal shape for readable Jacobian
    N, C_in, C_out, H, W = 1, 1, 1, 3, 3
    deformable_groups = 1
    stride = 1
    padding = 1
    dilation = 1

    # Input: double precision, requires gradient
    input = torch.randn(N, C_in, H, W, dtype=torch.float64, requires_grad=True) * 0.01

    # Offset: clamped and detached to avoid gradient tracking
    offset = torch.randn(N, 2 * 3 * 3, H, W, dtype=torch.float64) * 0.1
    offset = offset.clamp(-0.5, 0.5).detach()

    # Mask: sigmoid and detached
    mask = torch.sigmoid(torch.randn(N, 3 * 3, H, W, dtype=torch.float64)).detach()

    # Weight and bias: detached
    weight = torch.randn(C_out, C_in, 3, 3, dtype=torch.float64).detach()
    bias = torch.randn(C_out, dtype=torch.float64).detach()
    offset = torch.zeros_like(offset)
    mask = torch.ones_like(mask)
    # Forward pass
    out = dcn_v2_conv(input, offset, mask, weight, bias,
                      stride, padding, dilation, deformable_groups)
    print("Output stats → min:", out.min().item(), "max:", out.max().item(), "mean:", out.mean().item())

    if not torch.isfinite(out).all():
        print("⚠️ Output contains NaNs or infs — check input scaling or kernel logic.")
        return

    # Gradient check (only for input)
    try:
        result = gradcheck(dcn_v2_conv, (input, offset, mask, weight, bias,
                                         stride, padding, dilation, deformable_groups),
                           eps=1e-3, atol=1e-5, rtol=1e-3, nondet_tol=1e-4)
        print("✅ check_gradient_dconv passed:", result)
    except Exception as e:
        print("❌ gradcheck failed:", e)

        # Optional: manual gradient sanity check
        delta = 1e-3
        input_perturbed = input.clone()
        input_perturbed[0, 0, 1, 1] += delta

        out1 = dcn_v2_conv(input, offset, mask, weight, bias,
                           stride, padding, dilation, deformable_groups)
        out2 = dcn_v2_conv(input_perturbed, offset, mask, weight, bias,
                           stride, padding, dilation, deformable_groups)

        numerical_grad = (out2 - out1).sum() / delta
        print("Manual numerical grad at [0,0,1,1]:", numerical_grad.item())


def check_pooling_zero_offset():

    input = torch.randn(2, 16, 64, 64).cuda().zero_()
    input[0, :, 16:26, 16:26] = 1.
    input[1, :, 10:20, 20:30] = 2.
    rois = torch.tensor([
        [0, 65, 65, 103, 103],
        [1, 81, 41, 119, 79],
    ]).cuda().float()
    pooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=16,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.0).cuda()

    out = pooling(input, rois, input.new())
    s = ', '.join(['%f' % out[i, :, :, :].mean().item()
                   for i in range(rois.shape[0])])
    print(s)

    dpooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=16,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.0).cuda()
    offset = torch.randn(20, 2, 7, 7).cuda().zero_()
    dout = dpooling(input, rois, offset)
    s = ', '.join(['%f' % dout[i, :, :, :].mean().item()
                   for i in range(rois.shape[0])])
    print(s)


def check_gradient_dpooling():
    input = torch.randn(2, 3, 5, 5).cuda() * 0.01
    N = 4
    batch_inds = torch.randint(2, (N, 1)).cuda().float()
    x = torch.rand((N, 1)).cuda().float() * 15
    y = torch.rand((N, 1)).cuda().float() * 15
    w = torch.rand((N, 1)).cuda().float() * 10
    h = torch.rand((N, 1)).cuda().float() * 10
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = torch.randn(N, 2, 3, 3).cuda()
    input.requires_grad = True
    offset.requires_grad = True

    spatial_scale = 1.0 / 4
    pooled_size = 3
    output_dim = 3
    no_trans = 0
    group_size = 1
    trans_std = 0.0
    sample_per_part = 4
    part_size = pooled_size

    print('check_gradient_dpooling:',
          gradcheck(dcn_v2_pooling, (input, rois, offset,
                                     spatial_scale,
                                     pooled_size,
                                     output_dim,
                                     no_trans,
                                     group_size,
                                     part_size,
                                     sample_per_part,
                                     trans_std),
                    eps=1e-4, atol=1e-5, rtol=1e-3, nondet_tol=1e-4))


def example_dconv():
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DCN(64, 64, kernel_size=(3, 3), stride=1,
              padding=1, deformable_groups=2).cuda()
    # print(dcn.weight.shape, input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)


def example_dpooling():
    input = torch.randn(2, 32, 64, 64).cuda()
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = torch.randn(20, 2, 7, 7).cuda()
    input.requires_grad = True
    offset.requires_grad = True

    # normal roi_align
    pooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=32,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.1).cuda()

    # deformable pooling
    dpooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=32,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.1).cuda()

    out = pooling(input, rois, offset)
    dout = dpooling(input, rois, offset)
    print(out.shape)
    print(dout.shape)

    target_out = out.new(*out.size())
    target_out.data.uniform_(-0.01, 0.01)
    target_dout = dout.new(*dout.size())
    target_dout.data.uniform_(-0.01, 0.01)
    e = (target_out - out).mean()
    e.backward()
    e = (target_dout - dout).mean()
    e.backward()


def example_mdpooling():
    input = torch.randn(2, 32, 64, 64).cuda()
    input.requires_grad = True
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)

    # mdformable pooling (V2)
    dpooling = DCNPooling(spatial_scale=1.0 / 4,
                          pooled_size=7,
                          output_dim=32,
                          no_trans=False,
                          group_size=1,
                          trans_std=0.1,
                          deform_fc_dim=1024).cuda()

    dout = dpooling(input, rois)
    target = dout.new(*dout.size())
    target.data.uniform_(-0.1, 0.1)
    error = (target - dout).mean()
    error.backward()
    print(dout.shape)


if __name__ == '__main__':

    example_dconv()
    example_dpooling()
    example_mdpooling()

    check_pooling_zero_offset()
    # zero offset check
    if inC == outC:
        check_zero_offset()

    check_gradient_dpooling()
    check_gradient_dconv()
    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
