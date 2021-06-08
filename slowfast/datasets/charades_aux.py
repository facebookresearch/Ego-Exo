#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import random
from itertools import chain as chain
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging

from . import utils as utils
from .build import DATASET_REGISTRY
from .charades import Charades
from .kinetics_aux import _load, get_predicts, parse_handobj, spatial_sampling_with_boxes

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Charadeshandobj(Charades):
    """
    Charades loader with loading hand-obj boxes
    """
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
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
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        num_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        video_length = len(self._path_to_videos[index])
        assert video_length == len(self._labels[index])

        clip_length = (num_frames - 1) * sampling_rate + 1
        if temporal_sample_index == -1:
            if clip_length > video_length:
                start = random.randint(video_length - clip_length, 0)
            else:
                start = random.randint(0, video_length - clip_length)
        else:
            gap = float(max(video_length - clip_length, 0)) / (
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1
            )
            start = int(round(gap * temporal_sample_index))

        seq = [
            max(min(start + i * sampling_rate, video_length - 1), 0)
            for i in range(num_frames)
        ]
        frames = torch.as_tensor(
            utils.retry_load_images(
                [self._path_to_videos[index][frame] for frame in seq],
                self._num_retries,
            )
        )

        label = utils.aggregate_labels(
            [self._labels[index][i] for i in range(seq[0], seq[-1] + 1)]
        )
        label = torch.as_tensor(
            utils.as_binary_vector(label, self.cfg.MODEL.NUM_CLASSES)
        )

        # hand, objects
        frame_index = seq
        videoname = self._path_to_videos[index][0]
        videoname = videoname[videoname.rfind("/") + 1:videoname.rfind("-")]
        #hands, objects = parse_handobj(self.handobjs[videoname], frame_index)
        _, data = _load(videoname, self.cfg.HANDOBJ.DETS_FOLDER)
        hands, objects = parse_handobj(data, frame_index)
        assert len(data) == video_length, f"{len(data)} vs nf {video_length}"

        num_hand = hands.shape[0]
        boxes = np.concatenate((hands, objects), axis=0)

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
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
        boxes[:, :4] = boxes[:, :4] / crop_size
        hands = boxes[:num_hand, :]
        objects = boxes[num_hand:, :]
        # print(hands, objects)
        meta = {
            "hands": torch.from_numpy(hands).float(),
            "objs": torch.from_numpy(objects).float(),
        }

        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, label, index, meta
