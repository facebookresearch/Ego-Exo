#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
from torch.nn import functional as F

import slowfast.utils.metrics as metrics
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY
from .handobj_models import HandObjHead, HandObjHeadSingle, HandObjSlowFast
from .multi_models import MultiTaskHead, MultiTaskResNet, MultiTaskSlowFast
from .video_model_builder import _POOL1, ResNet, SlowFast


@MODEL_REGISTRY.register()
class HandObjMultiTaskResNet(MultiTaskResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        hweight = cfg.HANDOBJ.HAND_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA
        oweight = cfg.HANDOBJ.OBJ_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA

        head_class = HandObjHead
        head = head_class(
            dim_in=[width_per_group * 32],
            conv_dims=[cfg.HANDOBJ.HEAD_CONV_DIM] * cfg.HANDOBJ.HEAD_NUM_CONV,
            loss_weight=(hweight, oweight),
            with_targets=cfg.TEST.HEAD_TARGETS,
            loss_mode=cfg.HANDOBJ.LOSS_MODE,
            pos_weight=cfg.HANDOBJ.POS_WEIGHT,
        )
        self.hohead_name = "handobjhead" #"gazehead{}".format(cfg.TASK)
        self.add_module(self.hohead_name, head)

        if cfg.HANDOBJ.FREEZE_HANDOBJ_HEAD:
            print("freeze handobjhead")
            for p in self.handobjhead.parameters():
                p.requires_grad = False
    
    def forward(self, x, meta=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        feat = x

        head = getattr(self, self.head_name)
        x, extra_loss = head(feat)

        hohead = getattr(self, self.hohead_name)
        ho_logits, loss = hohead(feat, meta)

        loss.update(extra_loss)
        # print(gaze_logits.shape)

        return x + ho_logits, loss


@MODEL_REGISTRY.register()
class HandObjMultiTaskSlowFastHSlow(HandObjSlowFast):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]
        hweight = cfg.HANDOBJ.HAND_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA
        oweight = cfg.HANDOBJ.OBJ_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA

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


        head_class = HandObjHeadSingle
        head = head_class(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            conv_dims=[cfg.HANDOBJ.HEAD_CONV_DIM] * cfg.HANDOBJ.HEAD_NUM_CONV,
            loss_weight=(hweight, oweight),
            with_targets=cfg.TEST.HEAD_TARGETS,
            loss_mode=cfg.HANDOBJ.LOSS_MODE,
            pos_weight=cfg.HANDOBJ.POS_WEIGHT,
            pathway=0,
        )
        self.hohead_name = "handobjhead" #"gazehead{}".format(cfg.TASK)
        self.add_module(self.hohead_name, head)

        if cfg.HANDOBJ.FREEZE_HANDOBJ_HEAD:
            print("freeze handobjhead")
            for p in self.handobjhead.parameters():
                p.requires_grad = False
    
    def forward(self, x, meta=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        feat = x

        head = getattr(self, self.head_name)
        x, extra_loss = head(feat)

        hohead = getattr(self, self.hohead_name)
        ho_logits, loss = hohead(feat, meta)

        loss.update(extra_loss)
        # print(gaze_logits.shape)

        return x + ho_logits, loss
