# Ego-Exo

This repository is an official pytorch implementation of Ego-Exo work:

**[Ego-Exo: Transferring Visual Representations from Third-person to First-person Videos](https://arxiv.org/abs/2104.07905)**, CVPR, 2021

*Yanghao Li, Tushar Nagarajan, Bo Xiong and Kristen Grauman.*  


## Requirements

### Installation
Please follow the instructions in [INSTALL.md](INSTALL.md).

### Datasets
Please follow the descriptions in [DATASET.md](slowfast/datasets/DATASET.md) for dataset preparation.

### Pre-computed Ego-Exo scores for auxiliary tasks

We provided the pre-computed Ego-Score, Object-Score and Interaction-Map for auxiliary tasks. Please check the paper for the details about the Ego-Exo scores on auxiliary tasks, 

1. [optional] **Kinetics**.
We already provide the pre-trained Kinetics models, so you don't need the Ego-Exo scores of auxiliary tasks on Kinetics unless you want to train Ego-Exo from scratch on Kinetics. Please download the [Ego-Score](https://dl.fbaipublicfiles.com/ego-exo/aux_predictions/k400_ego_predicts.pkl), [Object-Score](https://dl.fbaipublicfiles.com/ego-exo/aux_predictions/k400_IN_predicts.pkl) and [Interaction-Map](https://dl.fbaipublicfiles.com/ego-exo/aux_predictions/k400_handobj_predicts.zip) on Kinetics, respectively.

2. [optional] **Ego datasets**.
For Charades-Ego, Epic-55 and Epic-100 datasets, we need Interaction-Map for Ego-Exo* models. Please find the pre-computed Interaction-Map for [Charades-Ego](https://dl.fbaipublicfiles.com/ego-exo/aux_predictions/charades_ego_handobj_predicts.zip), [Epic-55](https://dl.fbaipublicfiles.com/ego-exo/aux_predictions/epic-55_handobj_predicts.zip) and [Epic-100](https://dl.fbaipublicfiles.com/ego-exo/aux_predictions/epic-100_handobj_predicts.zip), respectively.

After downloading the Ego-Exo score files, we can put them under `data/aux_predictions/`.

## Pre-trained Kinetics models
We provide pre-trained models on Kinetics for both baseline and Ego-Exo.
* [SLOW_8x8_R50.pyth](https://dl.fbaipublicfiles.com/ego-exo/pretrain_models/k400/SLOW_8x8_R50.pyth) and [Ego_Exo_SLOW_8x8_R50.pyth](https://dl.fbaipublicfiles.com/ego-exo/pretrain_models/k400/Ego_Exo_SLOW_8x8_R50.pyth)
* [SLOWFAST_8x8_R50.pyth](https://dl.fbaipublicfiles.com/ego-exo/pretrain_models/k400/SLOWFAST_8x8_R50.pyth) and [Ego_Exo_SLOWFAST_8x8_R50.pyth](https://dl.fbaipublicfiles.com/ego-exo/pretrain_models/k400/Ego_Exo_SLOWFAST_8x8_R50.pyth)
* [SLOWFAST_8x8_R101.pyth](https://dl.fbaipublicfiles.com/ego-exo/pretrain_models/k400/SLOWFAST_8x8_R101.pyth) and [Ego_Exo_SLOWFAST_8x8_R101.pyth](https://dl.fbaipublicfiles.com/ego-exo/pretrain_models/k400/Ego_Exo_SLOWFAST_8x8_R101.pyth)


## Train/evaluate on Egocentric datasets

Training Ego-Exo and Ego-Exo* models on the downstream egocentric datasets can be done using the following command:

```
python tools/run_net.py \
    --cfg configs/ego-exo/charades-ego/Ego_Exo_SLOWFAST_8x8_R50.yaml \
    TRAIN.CHECKPOINT_FILE_PATH  /path/to/kinetics/models/Ego_Exo_SLOWFAST_8x8_R50.pyth \
```

We use 8 GPUs to train Charades-Ego and Epic-Kitchen-55, and use 8x2 GPUs for Epic-Kitchen-100. You can specify different pre-trained weights and Ego-Exo and Ego-Exo* configs (under `configs/ego-exo/`) on different datasets.


## Results on Egocentric datasets

The results of different methods on the validation set are shown as bellow: 

### Charades-Ego
| Method                          | mAP   |
| ------------------------------- | ----- |
| Slow\_8x8\_R50                  | 24.73 |
| Ego\_Exo, Slow\_8x8\_R50        | 26.64 |
| SlowFast\_8x8\_R50              | 26.27 |
| Ego\_Exo, SlowFast\_8x8\_R50    | 28.03 |
| Ego\_Exo\*, SlowFast\_8x8\_R50  | 29.11 |
| SlowFast\_8x8\_R101             | 27.47 |
| Ego\_Exo, SlowFast\_8x8\_R101   | 28.69 |
| Ego\_Exo\*, SlowFast\_8x8\_R101 | 30.13 |

### Validation set on Epic-Kitchen-55

| Method                         | verb-top1 | verb-top5 | noun-top1 | noun-top5 |
| ------------------------------ | --------- | --------- | --------- | --------- |
| SlowFast\_8x8\_R50             | 64.05     | 88.49     | 48.58     | 70.06     |
| Ego\_Exo, SlowFast\_8x8\_R50   | 65.97     | 88.91     | 49.42     | 72.35     |
| Ego\_Exo\*, SlowFast\_8x8\_R50 | 66.43     | 89.16     | 49.79     | 71.6      |

### Validation set on Epic-Kitchen-100

| Method                          | verb-top1 | verb-top5 | noun-top1 | noun-top5 |
| ------------------------------- | --------- | --------- | --------- | --------- |
| SlowFast\_8x8\_R50              | 66.02     | 90.13     | 50.51     | 75.02     |
| Ego\_Exo, SlowFast\_8x8\_R50    | 65.87     | 90.09     | 51.81     | 75.49     |
| Ego\_Exo\*, SlowFast\_8x8\_R50  | 66.77     | 90.3      | 52.03     | 76.25     |
| SlowFast\_8x8\_R101             | 65.41     | 90.25     | 50.97     | 75.43     |
| Ego\_Exo, SlowFast\_8x8\_R101   | 67.01     | 90.62     | 52.87     | 76.53     |
| Ego\_Exo\*, SlowFast\_8x8\_R101 | 67.48     | 90.66     | 52.93     | 76.83     |

For comparisons on the test set of Epic-Kitchen datasets, please check the results on the paper or submit results to the test server of EPIC-KITCHENS.

## License

The majority of this work is licensed under [CC-NC 4.0 International license](LICENSE). However, portions of the project are available under separate license terms: PySlowFast is licensed under the Apache 2.0 license.

## Cite

Ego-Exo is built on top of [PySlowFast](https://github.com/facebookresearch/SlowFast). If you find this repository useful in your own research, please consider citing:

```BibTeX
@inproceedings{ego-exo,
  title={Ego-Exo: Transferring Visual Representations from Third-person to First-person Videos},
  author={Li, Yanghao and Nagarajan, Tushar and Xiong, Bo and Grauman, Kristen},
  booktitle = {CVPR},
  year={2021}
}
```

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```