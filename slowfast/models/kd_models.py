#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY
from .video_model_builder import _POOL1, ResNet, SlowFast


class MultiTaskHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
    ):
        super(MultiTaskHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        projs = []
        for n in num_classes:
            projs.append(nn.Linear(sum(dim_in), n, bias=True))
        self.projections = nn.ModuleList(projs)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        feat = x
        if hasattr(self, "dropout"):
            feat = self.dropout(feat)

        x = []
        for projection in self.projections:
           # print(feat.shape, projection)
            x.append(projection(feat))

        # Performs fully convlutional inference.
        if not self.training:
            if not self.test_noact:
                x = [self.act(x_i) for x_i in x]
            x = [x_i.mean([1, 2, 3]) for x_i in x]

        x = [x_i.view(x_i.shape[0], -1) for x_i in x]
        return x, {}


@MODEL_REGISTRY.register()
class KDResNet(ResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]
        kd_num_classes = cfg.KD.NUM_CLASSES_LIST if cfg.KD.MULTI else [cfg.KD.NUM_CLASSES]

        head = MultiTaskHead(
            dim_in=[width_per_group * 32],
            num_classes=[cfg.MODEL.NUM_CLASSES] + kd_num_classes,
            pool_size=[None, None]
            if cfg.MULTIGRID.SHORT_CYCLE
            else [
                [
                    cfg.DATA.NUM_FRAMES // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )
        self.head_name = "head{}".format(cfg.TASK)
        self.add_module(self.head_name, head)



@MODEL_REGISTRY.register()
class KDSlowFast(SlowFast):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]
        kd_num_classes = cfg.KD.NUM_CLASSES_LIST if cfg.KD.MULTI else [cfg.KD.NUM_CLASSES]

        head = MultiTaskHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=[cfg.MODEL.NUM_CLASSES] + kd_num_classes,
            pool_size=[None, None]
            if cfg.MULTIGRID.SHORT_CYCLE
            else [
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )
        self.head_name = "head{}".format(cfg.TASK)
        self.add_module(self.head_name, head)
