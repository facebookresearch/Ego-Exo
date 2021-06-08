#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import functools
import numpy as np
import os
import pickle
import random
from multiprocessing import Manager, Pool
import torch
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .kinetics import Kinetics

logger = logging.get_logger(__name__)


@functools.lru_cache(maxsize=64)
def _load(videoname, folder):
    data = np.load(f"{folder}/{videoname}.npy", allow_pickle=True)

    tdata = []
    for d in data:
        o, h = d
        o = None if o is None else o[:, :6]
        h = None if h is None else h[:, :6]
        tdata.append((o, h))

    return (videoname, tdata)


def parse_handobj(handobjs, frame_index):
    def parse_dets(idx, dets):
        if dets is None:
            return None

        ret = dets[:, :6]
        ret[:, 5] = idx

        return ret

    hand, obj = [], []
    for i, idx in enumerate(frame_index):
        obj_dets, hand_dets = handobjs[idx]
        obj_dets = parse_dets(i, obj_dets)
        hand_dets = parse_dets(i, hand_dets)
        if obj_dets is not None:
            obj.append(obj_dets)
        if hand_dets is not None:
            hand.append(hand_dets)

    hand = np.concatenate(hand, axis=0) if len(hand) > 0 else np.zeros((0, 6))
    obj = np.concatenate(obj, axis=0) if len(obj) > 0 else np.zeros((0, 6))

    return hand, obj


def spatial_sampling_with_boxes(
    frames,
    boxes,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `channel` x `height` x `width`..
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    ori_boxes = boxes.copy()
    boxes = boxes[:, :4]
    if spatial_idx == -1:
        frames, boxes = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
            boxes=boxes,
        )
        # print("scale", frames.shape, boxes)
        frames, boxes = transform.random_crop(frames, crop_size, boxes=boxes)
        # print("crop", frames.shape, boxes)
        if random_horizontal_flip:
            frames, boxes = transform.horizontal_flip(0.5, frames, boxes=boxes)
        # print("flip", frames.shape, boxes)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, boxes = transform.random_short_side_scale_jitter(
            frames,
            min_scale,
            max_scale,
            boxes=boxes,
        )
        frames, boxes = transform.uniform_crop(frames,
                                               crop_size,
                                               spatial_idx,
                                               boxes=boxes)

    boxes = transform.clip_boxes_to_image(boxes, crop_size, crop_size)
    ori_boxes[:, :4] = boxes
    return frames, ori_boxes


def get_predicts(fpredict):
    if fpredict is None:
        return None
    with open(fpredict, "rb") as f:
        pred = pickle.load(f)

    return pred


@DATASET_REGISTRY.register()
class Kineticskd(Kinetics):
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            data_list = self.cfg.TRAIN.TRAIN_DATA_LIST
            teacher_path_list = [self.cfg.KD.TRAIN_TEACHER_PATH] \
                if not self.cfg.KD.MULTI else self.cfg.KD.TRAIN_TEACHER_PATH_LIST
        elif self.mode == "val":
            data_list = self.cfg.TRAIN.VAL_DATA_LIST
            teacher_path_list = None
        else:
            data_list = self.cfg.TEST.DATA_LIST
            teacher_path_list = None
    
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            data_list,
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        teacher_preds_list = [get_predicts(path) for path in teacher_path_list] \
            if teacher_path_list is not None else None

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._teacher_labels = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                teacher_label = [0] if teacher_preds_list is None else [teacher_preds[path] for teacher_preds in teacher_preds_list]
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._teacher_labels.append(teacher_label)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self.mode, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        frames, label, index, _ = super().__getitem__(index)
        meta = {"teacher_label": self._teacher_labels[index]}

        return frames, label, index, meta


@DATASET_REGISTRY.register()
class Kineticshandobj(Kinetics):
    """
    Kinetics video loader with loading hand objects boxes.
    online load handobj
    """
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S))
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale) * crop_size /
                        self.cfg.MULTIGRID.DEFAULT_S))
        elif self.mode in ["test"]:
            temporal_sample_index = (self._spatial_temporal_idx[index] //
                                     self.cfg.TEST.NUM_SPATIAL_CROPS)
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (self._spatial_temporal_idx[index] %
                                    self.cfg.TEST.NUM_SPATIAL_CROPS)
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE
                                               ] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(
                self.mode))
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e))
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames, frame_index = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=max_scale,
                ret_index=True,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # hand, objects
            videoname = os.path.basename(self._path_to_videos[index])
            videoname = videoname[:videoname.rfind(".")]
            #hands, objects = parse_handobj(self.handobjs[videoname], frame_index)
            _, data = _load(videoname, self.cfg.HANDOBJ.DETS_FOLDER)
            hands, objects = parse_handobj(data, frame_index)

            num_hand = hands.shape[0]
            boxes = np.concatenate((hands, objects), axis=0)

            # Perform color normalization.
            frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN,
                                            self.cfg.DATA.STD)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            # print("before", boxes)
            frames, boxes = spatial_sampling_with_boxes(
                frames,
                boxes,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            # print(frames.shape, boxes)
            boxes[:, :4] = boxes[:, :4] / crop_size
            hands = boxes[:num_hand, :]
            objects = boxes[num_hand:, :]

            meta = {
                "hands": torch.from_numpy(hands).float(),
                "objs": torch.from_numpy(objects).float(),
            }

            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, label, index, meta
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries))


@DATASET_REGISTRY.register()
class Kineticshandobjkd(Kineticshandobj):
    """
    Kinetics video loader with loading hand objects boxes and kd teacher label.
    online load handobj
    """
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            data_list = self.cfg.TRAIN.TRAIN_DATA_LIST
            teacher_path_list = [self.cfg.KD.TRAIN_TEACHER_PATH] \
                if not self.cfg.KD.MULTI else self.cfg.KD.TRAIN_TEACHER_PATH_LIST
        elif self.mode == "val":
            data_list = self.cfg.TRAIN.VAL_DATA_LIST
            teacher_path_list = None
        else:
            data_list = self.cfg.TEST.DATA_LIST
            teacher_path_list = None

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            data_list,
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file)

        teacher_preds_list = [get_predicts(path) for path in teacher_path_list] \
            if teacher_path_list is not None else None

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._teacher_labels = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (len(
                    path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2)
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR)
                teacher_label = [0] if teacher_preds_list is None else [
                    teacher_preds[path] for teacher_preds in teacher_preds_list
                ]
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._teacher_labels.append(teacher_label)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (len(self._path_to_videos) >
                0), "Failed to load Kinetics split {} from {}".format(
                    self.mode, path_to_file)
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file))

    def __getitem__(self, index):
        frames, label, index, meta = super().__getitem__(index)
        meta["teacher_label"] = self._teacher_labels[index]

        return frames, label, index, meta
