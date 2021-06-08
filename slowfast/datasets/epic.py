#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
from itertools import chain as chain
import pandas as pd
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Epickitchen(torch.utils.data.Dataset):
    '''
    Support epic-55 and epic-100
    '''
    def __init__(self, cfg, mode, num_retries=10):
        """
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Epic ".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.is_epic100 = cfg.DATA.EPIC_100

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Epic {}...".format(mode))
        self._construct_loader()
    
    def load_annotations(self, path_file):
        data = pd.read_pickle(path_file)

        videos = []
        for tup in data.iterrows():
            series = tup[1]
            item = {
                "participant_id": series["participant_id"],
                "video_id": series["video_id"],
                "start_frame": series["start_frame"],
                "stop_frame": series['stop_frame'],
                "verb_label": series.get("verb_class", -1),
                "noun_label": series.get("noun_class", -1),
            }
            videos.append(item)
        
        return videos


    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            data_list = self.cfg.TRAIN.TRAIN_DATA_LIST
        elif self.mode == "val":
            data_list = self.cfg.TRAIN.VAL_DATA_LIST
        else:
            data_list = self.cfg.TEST.DATA_LIST
    
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            data_list,
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._videos = self.load_annotations(path_to_file)

        self._videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._videos]
            )
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [range(self._num_clips) for _ in range(len(self._videos))]
            )
        )

        logger.info(
            "Epic dataloader constructed (size: {}) from {}".format(
                len(self._videos), path_to_file
            )
        )
    
    def get_frame_path(self, frame, index):
        if self.is_epic100:
            return f"{self.cfg.DATA.PATH_PREFIX}/{self._videos[index]['participant_id']}" \
                f"/rgb_frames/{self._videos[index]['video_id']}/frame_{frame:010}.jpg"
        else:
            return f"{self.cfg.DATA.PATH_PREFIX}/{self._videos[index]['video_id']}" \
                f"/frame_{frame:010}.jpg"

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

        if self.mode in ["train"]:
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
        elif self.mode in ["val"]:
            temporal_sample_index = int(self.cfg.TEST.NUM_ENSEMBLE_VIEWS / 2)
            spatial_sample_index = 1

            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3

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
        start_frame, end_frame = int(self._videos[index]["start_frame"]), int(self._videos[index]["stop_frame"])
        video_length = end_frame - start_frame + 1

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
                [self.get_frame_path(frame + start_frame, index) for frame in seq],
                self._num_retries,
            )
        )

        label = torch.tensor((int(self._videos[index]["verb_label"]), int(self._videos[index]["noun_label"])))

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, label, index, {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._videos)
