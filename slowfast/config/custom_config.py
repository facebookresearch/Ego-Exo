#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Add your own customized configs.

    _C.TASK = ""

    # if using epic-55 or epic-100
    _C.DATA.EPIC_100 = False

    # multi-class
    _C.MODEL.MULTI_CLASS = False
    _C.MODEL.NUM_CLASSES_LIST = []
    
    # Train/val split files
    _C.TRAIN.TRAIN_DATA_LIST = "train.csv"
    _C.TRAIN.VAL_DATA_LIST = "val.csv"


    # Test split files
    _C.TEST.DATA_LIST = "test.csv"
    #  Test saving path
    _C.TEST.SAVE_PREDICT_PATH = "predicts.pkl"

    _C.TEST.NO_ACT = False

    _C.TEST.HEAD_TARGETS = True

    _C.TEST.LOAD_BEST = False

    # Model
    _C.MODEL.FREEZE_STAGES = 0

    _C.MODEL.FREEZE_BN = False

    # Knowledge Distill args
    _C.KD = CfgNode()

    _C.KD.ENABLE = False

    _C.KD.MULTI = False

    _C.KD.NUM_CLASSES = 2
    _C.KD.NUM_CLASSES_LIST = [2]

    # KD loss weights
    _C.KD.ALPHA_LIST = [0.5]
    _C.KD.ALPHA = 0.5

    # KD loss temporature param
    _C.KD.T_LIST = [1.0]
    _C.KD.T = 1.0

    # Teachers' predictions
    _C.KD.TRAIN_TEACHER_PATH = ""
    _C.KD.TRAIN_TEACHER_PATH_LIST = [""]
    _C.KD.TEACHER_CFG_PATH = ""
    _C.KD.TEACHER_CHECKPOINT_PATH = ""

    _C.KD.MODE = "none"

    _C.KD.BALANCE_LOSS = True # if multiply (1-alpha) to other loss

    _C.KD.SAMPLE_RATIO = 0.5


    # Loss
    _C.LOSS = CfgNode()
    # cross-entropy loss weight
    _C.LOSS.CE_LOSS_WEIGHT = 1.0

    # handobj head params
    _C.HANDOBJ = CfgNode()

    _C.HANDOBJ.ENABLE = False

    _C.HANDOBJ.DETS_FOLDER = "data/kinetics/288p_handobj_results"

    _C.HANDOBJ.HEAD_NUM_CONV = 1
    _C.HANDOBJ.HEAD_CONV_DIM = 512
    _C.HANDOBJ.HAND_LOSS_WEIGHT = 1.0
    _C.HANDOBJ.OBJ_LOSS_WEIGHT = 1.0
    _C.HANDOBJ.ALPHA = 1.0
    _C.HANDOBJ.BALANCE_LOSS = False
    _C.HANDOBJ.LOSS_MODE = "soft"
    _C.HANDOBJ.POS_WEIGHT = 1.0
    _C.HANDOBJ.KD_LOSS = False
    _C.HANDOBJ.FREEZE_HANDOBJ_HEAD = False
