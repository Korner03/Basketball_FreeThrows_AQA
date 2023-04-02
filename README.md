# Revisiting Pose Based AQA For Basketball Free Throws Prediction

We provide PyTorch implementation for our paper [_Revisiting Pose Based AQA For Basketball Free Throws Prediction_](https://motionretargeting2d.github.io/).

## Prerequisites

- Python 3.7
- PyTorch 1.9
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo

  ```bash
  git clone https://github.com/Korner03/Basketball_FreeThrows_AQA.git
  cd Basketball_FreeThrows_AQA
  ```

- Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

## Train from scratch

### Prepare Data

- Download BFTS

  Download bbfts_data from [Google Drive](https://drive.google.com/drive/folders/169b13uVy3mr-gs9WitMJ9fPdA82PzgSx?usp=sharing), then unzip into to ./bbfts_data.

  > NOTE: The data is fully prepared. To create the aligned motion sequence from scratch see [README](.data_prep/README.md) from data_prep directory.

- Download pretrained model

  Download 'pretrained_skeleton.pth' From checkpoints directory in the above link (taken from [link](https://github.com/ChrisWu1997/2D-Motion-Retargeting/tree/master/model) and place in ./checkpoints directory.

### Train & Test

- Train:

  ```
  python train.py --config configs/config.yaml
  ```

  The model will be saved under ./experiments/<exp-number>/model.pth

- Test:

  ```
  python test.py --config configs/config.yaml --ckpt experiments/<exp-number>/model.pth
  ```

## Credits

Credits
This application is inspired by Open Source components. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: [Learning Character-Agnostic Motion for Motion Retargeting in 2D](https://github.com/ChrisWu1997/2D-Motion-Retargeting)

Copyright (c) 2019 Rundi Wu

License (MIT): [LICENSE](https://github.com/ChrisWu1997/2D-Motion-Retargeting/blob/master/LICENSE)


## Citation
If you use this code for your research, please cite our paper:
```
@article{a,
  author = {b},
  title = {c},
  journal = {d},
  volume = {e},
  number = {f},
  pages = {g},
  year = {h},
  publisher = {i}
}

```