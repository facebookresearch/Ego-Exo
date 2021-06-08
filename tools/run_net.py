#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from tools.demo_net import demo
from tools.epic.handobj.test_net import test as test_epic_handobj
from tools.epic.handobj.train_net import train as train_epic_handobj
from tools.epic.test_net import test as test_epic
from tools.epic.train_net import train as train_epic
from tools.handobj.test_net import test as test_handobj
from tools.handobj.train_net import train as train_handobj
from tools.kd.test_net import test as test_kd
from tools.kd.train_net import train as train_kd
from tools.test_net import test
from tools.train_net import train
from tools.visualization import visualize


def get_func(cfg):
    train_func = train
    if cfg.TRAIN.DATASET in ["epickitchen", "epickitchenhandobj"]:
        if cfg.HANDOBJ.ENABLE:
            train_func = train_epic_handobj
        else:
            train_func = train_epic
    elif cfg.KD.ENABLE:
        train_func = train_kd
    elif cfg.HANDOBJ.ENABLE:
        train_func = train_handobj

    test_func = test
    if cfg.TEST.DATASET in ["epickitchen", "epickitchenhandobj"]:
        if cfg.HANDOBJ.ENABLE:
            test_func = test_epic_handobj
        else:
            test_func = test_epic
    elif cfg.KD.ENABLE:
        test_func = test_kd
    elif cfg.HANDOBJ.ENABLE:
        test_func = test_handobj

    return train_func, test_func

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    train_func, test_func = get_func(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train_func)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test_func)

    if cfg.DEMO.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=demo)

    if cfg.TENSORBOARD.ENABLE and cfg.TENSORBOARD.MODEL_VIS.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
