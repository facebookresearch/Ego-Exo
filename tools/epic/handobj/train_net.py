#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MODEL.FREEZE_STAGES > 0:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm3d) and name.startswith("module.s") and int(name[8]) <= cfg.MODEL.FREEZE_STAGES \
                and name[9] == ".":
                m.eval()
                logger.info(f"freeze bn layer {name}")

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

        else:
            # Perform the forward pass.
            preds, extra_loss = model(inputs, meta)
        
        verb_preds, noun_preds = preds[:2]
        verb_labels, noun_labels = labels[:, 0], labels[:, 1]
        
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        # Compute the loss.
        loss = loss_fun(verb_preds, verb_labels) + loss_fun(noun_preds, noun_labels)

        extra_loss["ce_loss"] = loss * cfg.LOSS.CE_LOSS_WEIGHT
        w_ce = 1.0
        if cfg.HANDOBJ.BALANCE_LOSS:
            w_ce -= cfg.HANDOBJ.ALPHA

        if cfg.HANDOBJ.KD_LOSS:
            kd_preds = ori_preds[2:-1]
            for i in range(len(kd_preds)):
                kd_loss = loss_fn_kd(
                    kd_preds[i],
                    meta["teacher_label"][i],
                    alpha=cfg.KD.ALPHA_LIST[i] if cfg.KD.MULTI else cfg.KD.ALPHA,
                    T=cfg.KD.T_LIST[i] if cfg.KD.MULTI else cfg.KD.T,
                )
                extra_loss[f"kd_loss_{i}"] = kd_loss
            # kd_loss = loss_fn_kd(preds1, meta["teacher_label"], alpha=cfg.KD.ALPHA, T=cfg.KD.T)
            
            if cfg.KD.BALANCE_LOSS:
                alphas = sum(cfg.KD.ALPHA_LIST) if cfg.KD.MULTI else cfg.KD.ALPHA
                w_ce -= alphas
            #     for key in extra_loss.keys():
            #         if "loss" in key:
            #             extra_loss[key] *= (1.0 - cfg.KD.ALPHA)
            #extra_loss["kd_loss"] = kd_loss

        extra_loss["ce_loss"] *= w_ce
        loss = 0
        for key in extra_loss.keys():
            if "loss" in key:
                loss += extra_loss[key]

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()


        verb_preds, noun_preds = preds[:2]
        verb_labels, noun_labels = labels[:, 0], labels[:, 1]
        # Compute the errors.
        ks = (1, 5)
        verb_num_topks_correct = metrics.topks_correct(verb_preds, verb_labels, ks)
        noun_num_topks_correct = metrics.topks_correct(noun_preds, noun_labels, ks)
        action_num_topks_correct = metrics.multitask_topks_correct(
            (verb_preds, noun_preds),
            (verb_labels, noun_labels),
            ks,
        )

        verb_top1_err, verb_top5_err  = [
            (1.0 - x / verb_preds.size(0)) * 100.0 for x in verb_num_topks_correct
        ]
        noun_top1_err, noun_top5_err  = [
            (1.0 - x / noun_preds.size(0)) * 100.0 for x in noun_num_topks_correct
        ]
        top1_err, top5_err = [
            (1.0 - x / verb_preds.size(0)) * 100.0 for x in action_num_topks_correct
        ]

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err, verb_top1_err, verb_top5_err, noun_top1_err, noun_top5_err = du.all_reduce(
                [loss, top1_err, top5_err, verb_top1_err, verb_top5_err, noun_top1_err, noun_top5_err]
            )

            for key in extra_loss.keys():
                extra_loss[key] = du.all_reduce([extra_loss[key]])[0]


        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err, verb_top1_err, verb_top5_err, noun_top1_err, noun_top5_err= (
            loss.item(),
            top1_err.item(), 
            top5_err.item(), 
            verb_top1_err.item(),
            verb_top5_err.item(),
            noun_top1_err.item(), 
            noun_top5_err.item(),
        )

        for key in extra_loss.keys():
            extra_loss[key] = extra_loss[key].item()

        stats = {
            "verb_top1_err": verb_top1_err,
            "verb_top5_err": verb_top5_err,
            "noun_top1_err": noun_top1_err,
            "noun_top5_err": noun_top5_err,
        }
        stats.update(extra_loss)

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            top1_err, top5_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS, stats=stats
        )
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        preds, _ = model(inputs, meta)

        verb_preds, noun_preds = preds[:2]
        verb_labels, noun_labels = labels[:, 0], labels[:, 1]
        # Compute the errors.
        ks = (1, 5)
        verb_num_topks_correct = metrics.topks_correct(verb_preds, verb_labels, ks)
        noun_num_topks_correct = metrics.topks_correct(noun_preds, noun_labels, ks)
        action_num_topks_correct = metrics.multitask_topks_correct(
            (verb_preds, noun_preds),
            (verb_labels, noun_labels),
            ks,
        )
        verb_top1_err, verb_top5_err  = [
            (1.0 - x / verb_preds.size(0)) * 100.0 for x in verb_num_topks_correct
        ]
        noun_top1_err, noun_top5_err  = [
            (1.0 - x / noun_preds.size(0)) * 100.0 for x in noun_num_topks_correct
        ]
        top1_err, top5_err  = [
            (1.0 - x / verb_preds.size(0)) * 100.0 for x in action_num_topks_correct
        ]
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err, verb_top1_err, verb_top5_err, noun_top1_err, noun_top5_err = du.all_reduce(
                [top1_err, top5_err, verb_top1_err, verb_top5_err, noun_top1_err, noun_top5_err]
            )

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err, verb_top1_err, verb_top5_err, noun_top1_err, noun_top5_err= (
            top1_err.item(), 
            top5_err.item(), 
            verb_top1_err.item(),
            verb_top5_err.item(),
            noun_top1_err.item(), 
            noun_top5_err.item(),
        )

        stats = {
            "verb_top1_err": verb_top1_err,
            "verb_top5_err": verb_top5_err,
            "noun_top1_err": noun_top1_err,
            "noun_top5_err": noun_top5_err,
        }

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            top1_err, top5_err, inputs[0].size(0) * cfg.NUM_GPUS, stats=stats
        )
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                global_step=len(val_loader) * cur_epoch + cur_iter,
            )

        val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.

    val_meter.reset()

    return is_best_epoch


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        try:
            misc.log_model_info(model, cfg, use_train_input=True)
        except Exception as e:
            logger.info(e)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
        )

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        ):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        ):
            is_best_epoch = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
            if is_best_epoch:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, is_best_epoch=True)

    if writer is not None:
        writer.close()
