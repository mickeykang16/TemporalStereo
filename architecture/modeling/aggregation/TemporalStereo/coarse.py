import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from architecture.modeling.layers import Conv2d, Conv3d
from architecture.modeling.aggregation.utils import block_cost

from .module import (ResidualBlock3D, DepthwiseConv3D,
                     ConvexUpsample, PyramidFusion, PredictionHeads)

class CoarseAggregation(nn.Module):
    def __init__(self,
                 in_planes: int,
                 C: int,
                 num_sample: int,
                 delta: float = 1,
                 block_cost_scale: int = 3,
                 topk: int = 2,
                 spatial_fusion: bool = True,
                 norm: str = 'BN3d',
                 activation: Union[str, Tuple, List] = 'SiLU'):
        super(CoarseAggregation, self).__init__()
        self.in_planes = in_planes
        self.C = C
        self.num_sample = num_sample
        self.delta = delta
        self.block_cost_scale = block_cost_scale
        self.norm = norm
        self.activation = activation
        self.spatial_fusion = spatial_fusion
        self.topk = topk

        cost_planes = 1 * in_planes + block_cost_scale * in_planes // 8
        self.init3d = nn.Sequential(
            DepthwiseConv3D(cost_planes, C, 3, 1, 1, bias=True, norm=norm, activation=activation),
            ResidualBlock3D(in_planes=C, kernel_size=3, stride=2, padding=1, norm=norm, activation=activation),
            DepthwiseConv3D(C, C, 3, 1, padding=2, dilation=2, bias=False, norm=norm, activation=activation),
        )

        self.past_conv = Conv3d(1, C, 1, 1, 0, bias=False, norm=(norm, C), activation=activation)
        if spatial_fusion:
            self.fuse = PyramidFusion(in_planes=C, norm=norm, activation=activation)

        self.pred_heads = PredictionHeads(in_planes=C, delta=delta, norm=norm, activation=activation)

        self.convex_upsample = ConvexUpsample(in_planes=in_planes, upscale_factor=2, window_size=3)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def predict_disp(self, cost, disp_sample, off, k=2):
        topk_cost, indices = torch.topk(cost, k=k, dim=1)
        prob = torch.softmax(topk_cost, dim=1)
        topk_disp = torch.gather(disp_sample+off, dim=1, index=indices)
        disp_map = torch.sum(prob*topk_disp, dim=1, keepdim=True)

        return disp_map, topk_disp, topk_cost

    def forward(self, left, right, prev_info:dict):
        B, _, H, W = left.shape
        raw_cost = block_cost(left, right, self.num_sample, block_cost_scale=self.block_cost_scale)
        disp_sample = torch.linspace(0, self.num_sample-1, self.num_sample).view(1, self.num_sample, 1, 1)
        disp_sample = disp_sample.expand(B, self.num_sample, H, W).to(left.device)
        init_cost = self.init3d(raw_cost)
        memory = prev_info.get('cost_memory', None)
        use_past_cost = prev_info.get('use_past_cost', False)
        if memory is None or not use_past_cost:
            memory_sample = torch.zeros_like(disp_sample[:, :self.topk])
            memory_volume = torch.zeros_like(memory_sample).unsqueeze(dim=1)

        else:
            memory_sample = memory['disp_sample']
            mh, mw = memory_sample.shape[-2:]
            memory_sample = F.interpolate(memory_sample*W/mw, size=(H, W), mode='bilinear', align_corners=True)
            # B x topK x h x w
            memory_volume = memory['cost_volume']
            memory_volume = F.interpolate(memory_volume, size=(H, W), mode='bilinear', align_corners=True)
            memory_volume = memory_volume.unsqueeze(dim=1)
        # B x C x topK x h x w
        memory_volume = self.past_conv(memory_volume)
        # [B, D, H, W]
        disp_sample = torch.cat([disp_sample, memory_sample], dim=1)
        # [B, C, 2*D, H, W]
        init_cost = torch.cat([init_cost, memory_volume], dim=2)
        disp_sample, indices = torch.sort(disp_sample, dim=1)
        init_cost = torch.gather(init_cost, dim=2, index=indices.unsqueeze(dim=1).repeat(1, self.C, 1, 1, 1))
        init_cost = init_cost.contiguous()

        if self.spatial_fusion:
            init_cost = self.fuse(init_cost)

        final_cost, off = self.pred_heads(init_cost)

        # learn disparity
        disp, memory_sample, memory_volume = self.predict_disp(final_cost, disp_sample, off, k=self.topk)
        disp = self.convex_upsample(left, disp)

        return disp, final_cost, off, disp_sample, prev_info
