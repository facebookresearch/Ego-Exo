#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


def loss_fn_kd(outputs, teacher_outputs, alpha=0.0, T=1.0, weight=None, mode="none", cfg=None):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    if mode == "none":
        KL_loss = nn.KLDivLoss(reduction="batchmean",)(
            F.log_softmax(outputs/T, dim=-1),
            F.softmax(teacher_outputs/T, dim=-1),
        ) * (alpha * T * T) 
    else:
        KL_loss = nn.KLDivLoss(reduction="none",)(
            F.log_softmax(outputs/T, dim=-1),
            F.softmax(teacher_outputs/T, dim=-1),
        ) * (alpha * T * T) 
        if mode in ["ego", "nonego"] :
            assert weight is not None
            # print(KL_loss.shape, weight.shape)
            w = F.softmax(weight, dim=-1)
            w = w[:, 1] if mode == "ego" else w[:, 0]
            # w[:] = 1.0
            KL_loss = KL_loss * w.view(-1, 1)
            KL_loss = torch.sum(KL_loss) / torch.sum(w)
        elif mode in ["sample_ego", "sample_nonego"]:
            w = F.softmax(weight, dim=-1)
            w = w[:, 1] if mode == "sample_ego" else w[:, 0]

            n = int(cfg.KD.SAMPLE_RATIO * outputs.shape[0])
            _, idx = torch.topk(w, k=n)

            KL_loss = torch.sum(KL_loss[idx, :]) / n
        elif mode in ["sample_entropy"]:
            w = F.softmax(teacher_outputs/T, dim=-1)
            entropy = torch.distributions.Categorical(w).entropy()

            n = int(cfg.KD.SAMPLE_RATIO * outputs.shape[0])
            _, idx = torch.topk(entropy, k=n, largest=False)

            KL_loss = torch.sum(KL_loss[idx, :]) / n
        else:
            raise NotImplementedError

    return KL_loss
