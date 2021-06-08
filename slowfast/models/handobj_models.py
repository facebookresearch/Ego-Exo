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
from .kd_models import MultiTaskHead
from .video_model_builder import _POOL1, ResNet, SlowFast


def gen_targets(gts, B, T, size, mode):
    assert mode in ["soft", "hard"]
    targets = torch.zeros((B, T, size, size)).cuda()
    # print(gts)
    for idx, gt in enumerate(gts):
        boxes = gt[:, :4] * size
        boxes = boxes.floor().long()
        boxes = torch.clamp(boxes, min=0, max=size - 1)

        for i in range(boxes.shape[0]):
            iT = int(gt[i, 5])
            # print(iT)
            x1, y1, x2, y2 = [x.item() for x in boxes[i, :4]]
            # x1, y1, x2, y2 = boxes[i, 0].item()
            if mode == 'soft':
                targets[idx, iT, y1:y2 + 1, x1:x2 + 1] = torch.clamp(
                    targets[idx, iT, y1:y2 + 1, x1:x2 + 1],
                    min=gt[i, 4].item(),
                )
            else:
                targets[idx, iT, y1:y2 + 1, x1:x2 + 1] = 1.0
    
    # print(targets[0, 0])
    
    return targets


def cal_handobj_loss(x, hand_gt, obj_gt, heatmap_size, pos_weight=1.0, mode="soft", pathway=-1):
    x_fast = x[0] if len(x) == 1 else x[1]
    B, _, T, H, W = x_fast.shape 
    # B x T x heatmap_size x heatmap_size
    hand_targets = gen_targets(hand_gt, B, T, heatmap_size, mode)
    obj_targets = gen_targets(obj_gt, B, T, heatmap_size, mode)
    if len(x) == 1:
        x = x[0]
    else:
        slow_T = x[0].shape[2]
        index = torch.linspace(0, T - 1, slow_T).long().cuda()
        slow_hand_targets = torch.index_select(hand_targets, 1, index)
        slow_obj_targets = torch.index_select(obj_targets, 1, index)

        if pathway == -1:
            # print(index, index.shape)
            x = torch.cat(x, dim=2)
            # print(x.shape)
            hand_targets = torch.cat((slow_hand_targets, hand_targets), dim=1)
            # print(hand_targets.shape)
            obj_targets = torch.cat((slow_obj_targets, obj_targets), dim=1)
        else:
            x = x[pathway]
            if pathway == 0:
                hand_targets = slow_hand_targets
                obj_targets = slow_obj_targets
        B, _, T, H, W = x.shape

    hand_targets = hand_targets.view(B * T, H * W)
    obj_targets = obj_targets.view(B * T, H * W)

    xhand = x[:, 0, :, :, :].reshape(B * T, H * W)
    xobj = x[:, 1, :, :, :].reshape(B * T, H * W)
    pos_weight_tensor = torch.ones([H * W]).cuda() * pos_weight
    loss_fun = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_tensor)
    loss = {
        "hand_loss": loss_fun(xhand, hand_targets),
        "obj_loss": loss_fun(xobj, obj_targets),
    }

    return loss


class HandObjHead(nn.Module):
    def __init__(
        self,
        dim_in,
        conv_dims=(512, ),
        loss_weight=(1.0, 0.0),
        with_targets=True,
        loss_mode="soft",
        pos_weight=1.0,
    ):
        super(HandObjHead, self).__init__()
        self.loss_weight = loss_weight
        self.with_targets = with_targets
        self.loss_mode = loss_mode
        self.pos_weight = pos_weight

        self.num_pathways = len(dim_in)
        self.build_blocks(dim_in, conv_dims)

    def build_blocks(self, dim_in, conv_dims):
        self.blocks = []
        for pathway in range(self.num_pathways):
            self.blocks.append([])
            in_channels = dim_in[pathway]
            for idx, layer_channels in enumerate(conv_dims):
                module = nn.Conv3d(in_channels, layer_channels, (1, 3, 3), stride=1, padding=(0, 1, 1))
                self.add_module("pathway{}_conv_fcn{}".format(pathway, idx), module)
                self.blocks[pathway].append(module)
                in_channels = layer_channels

        self.score_projection = nn.Conv3d(
            in_channels, 2, (1, 1, 1), stride=(1, 1, 1)
        )

    def forward(self, inputs, meta=None):
        x = inputs
        for pathway in range(self.num_pathways):
            for layer in self.blocks[pathway]:
                x[pathway] = F.relu(layer(x[pathway]))
        
            x[pathway] = self.score_projection(x[pathway])



        if self.with_targets:      
            hand_gt = meta["hands"] 
            obj_gt = meta["objs"]
            # print(x[0].shape, x[1].shape)
            #x = x[0]
            _, _, _, H, W = x[0].shape
            assert H == W
            #x = [x_i.reshape(B, C, T, H * W) for x_i in x]
            loss = cal_handobj_loss(x, hand_gt, obj_gt, H, pos_weight=self.pos_weight, mode=self.loss_mode)
            loss["hand_loss"] *= self.loss_weight[0]
            loss["obj_loss"] *= self.loss_weight[1]
        else:
            loss = {}

        return x, loss


@MODEL_REGISTRY.register()
class HandObjResNet(ResNet):
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

        return (x, ho_logits), loss


@MODEL_REGISTRY.register()
class HandObjSlowFast(SlowFast):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        hweight = cfg.HANDOBJ.HAND_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA
        oweight = cfg.HANDOBJ.OBJ_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA

        head_class = HandObjHead
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

        return (x, ho_logits), loss


class HandObjHeadSingle(HandObjHead):
    def __init__(
        self,
        dim_in,
        conv_dims=(512, ),
        loss_weight=(1.0, 0.0),
        with_targets=True,
        loss_mode="soft",
        pos_weight=1.0,
        pathway=0,
    ):
        self.pathway = pathway
        super(HandObjHeadSingle, self).__init__(
            dim_in, conv_dims, loss_weight, with_targets, loss_mode, pos_weight
        )

    def build_blocks(self, dim_in, conv_dims):
        self.blocks = []
        
        pathway = self.pathway
        in_channels = dim_in[pathway]
        for idx, layer_channels in enumerate(conv_dims):
            module = nn.Conv3d(in_channels, layer_channels, (1, 3, 3), stride=1, padding=(0, 1, 1))
            self.add_module("pathway{}_conv_fcn{}".format(pathway, idx), module)
            self.blocks.append(module)
            in_channels = layer_channels

        self.score_projection = nn.Conv3d(
            in_channels, 2, (1, 1, 1), stride=(1, 1, 1)
        )

    def forward(self, inputs, meta=None):
        x = inputs
        pathway = self.pathway
        
        for layer in self.blocks:
            x[pathway] = F.relu(layer(x[pathway]))
    
        x[pathway] = self.score_projection(x[pathway])

        if self.with_targets:      
            hand_gt = meta["hands"] 
            obj_gt = meta["objs"]
            # print(x[0].shape, x[1].shape)
            #x = x[0]
            _, _, _, H, W = x[0].shape
            assert H == W
            #x = [x_i.reshape(B, C, T, H * W) for x_i in x]
            loss = cal_handobj_loss(
                x, hand_gt, obj_gt, H, pos_weight=self.pos_weight, mode=self.loss_mode, pathway=self.pathway
            )
            loss["hand_loss"] *= self.loss_weight[0]
            loss["obj_loss"] *= self.loss_weight[1]
        else:
            loss = {}

        return x, loss


@MODEL_REGISTRY.register()
class HandObjSlowFastHFast(HandObjSlowFast):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        hweight = cfg.HANDOBJ.HAND_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA
        oweight = cfg.HANDOBJ.OBJ_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA

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
            pathway=1,
        )
        self.hohead_name = "handobjhead" #"gazehead{}".format(cfg.TASK)
        self.add_module(self.hohead_name, head)

        if cfg.HANDOBJ.FREEZE_HANDOBJ_HEAD:
            print("freeze handobjhead")
            for p in self.handobjhead.parameters():
                p.requires_grad = False


@MODEL_REGISTRY.register()
class HandObjSlowFastHSlow(HandObjSlowFast):
    def _construct_network(self, cfg):
        super()._construct_network(cfg)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        hweight = cfg.HANDOBJ.HAND_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA
        oweight = cfg.HANDOBJ.OBJ_LOSS_WEIGHT * cfg.HANDOBJ.ALPHA

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
