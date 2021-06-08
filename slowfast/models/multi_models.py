#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY
from .kd_models import MultiTaskHead
from .video_model_builder import _POOL1, ResNet, SlowFast


@MODEL_REGISTRY.register()
class MultiTaskResNet(ResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = MultiTaskHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES_LIST,
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
class MultiTaskSlowFast(SlowFast):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = MultiTaskHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES_LIST,
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
