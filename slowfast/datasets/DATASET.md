# Dataset Preparation

## Kinetics

The Kinetics Dataset could be downloaded via the code released by ActivityNet:

1. Download the videos via the official [scripts](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).

2. After all the videos were downloaded, resize the video to the short edge size of 256, then prepare the csv files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv`. The format of the csv file is:

```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```

All the Kinetics models in the Model Zoo are trained and tested with the same data as [Non-local Network](https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md). For dataset specific issues, please reach out to the [dataset provider](https://deepmind.com/research/open-source/kinetics).


### Charades-Ego

1. Download RGB frames of Charades-Ego from the [offical website](https://prior.allenai.org/projects/charades-ego).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/ego-exo/dataset_split_files/charades_ego_split/train_1st.csv), [val](https://dl.fbaipublicfiles.com/ego-exo/dataset_split_files/charades_ego_split/test_1st.csv)).


Please set `DATA.PATH_TO_DATA_DIR` to point to the folder containing the frame lists, and `DATA.PATH_PREFIX` to the folder containing RGB frames. For example, we set the symlinks as follow:

```
mkdir -p data/charades-ego
ln -s /path/to/Charades-Ego/CharadesEgo_v1_rgb data/charades-ego/rgb
ln -s /path/to/Charades-Ego/charades_ego_split data/charades-ego/split
```

### Epic-Kitchen-55

1. Download the RGB frames from [EPIC Kitchens 55](https://epic-kitchens.github.io/2019).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/ego-exo/dataset_split_files/epic_55_split/EPIC_train_action_labels.pkl), [val](https://dl.fbaipublicfiles.com/ego-exo/dataset_split_files/epic_55_split/EPIC_val_action_labels.pkl), [trainval](https://dl.fbaipublicfiles.com/ego-exo/dataset_split_files/epic_55_split/EPIC_train_val_action_labels.pkl). Note that the train/val split is following [temporal-binding-network](https://github.com/ekazakos/temporal-binding-network)

Please set `DATA.PATH_TO_DATA_DIR` to point to the folder containing the frame lists, and `DATA.PATH_PREFIX` to the folder containing RGB frames. For example, we set the symlinks as follow:

```
mkdir -p data/epic-55/split
ln -s /path/to/epic-kitchen-55/rgb_extracted/train data/epic-55/train_rgb_frames
ln -s /path/to/ego-exo/dataset_split_files/epic_55_split/ data/epic-55/split
```

### Epic-Kitchen-100

1. Download the RGB frames and annotations from [EPIC Kitchens 100](https://github.com/epic-kitchens/epic-kitchens-100-annotations).

2. Please set `DATA.PATH_TO_DATA_DIR` to point to the folder containing the frame lists, and `DATA.PATH_PREFIX` to the folder containing RGB frames. For example, we set the symlinks as follow:

```
mkdir -p data/epic-100/
ln -s  /path/to/EPIC-KITCHENS-100 data/epic-100/dataset
ln -s /path/to/epic-kitchens-100-annotations data/epic-100/annotations
```
