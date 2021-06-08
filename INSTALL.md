# Installation

Clone the repo:
```
git clone https://github.com/facebookresearch/ego-exo
```

Install required packages:
```
conda env create -f environment.yaml
conda activate ego-exo-oss

# Install d2
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

# Install slowfast in ego-exo
python setup.py build develop
```
Code tested with Python 3.7, cuda 10.1, cudnn 7.6.5