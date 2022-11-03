[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)
[![pytorch](https://img.shields.io/badge/pytorch-1.6.0-%2523ee4c2c.svg)](https://pytorch.org/)
[![pytorch](https://img.shields.io/badge/pytorch-1.7.1-%2523ee4c2c.svg)](https://pytorch.org/)

Sparse DETR (ICLR'22)
========

By [Byungseok Roh](https://scholar.google.com/citations?user=H4VWYHwAAAAJ)\*,  [Jaewoong Shin](https://scholar.google.com/citations?user=i_o_95kAAAAJ)\*,  [Wuhyun Shin](https://scholar.google.com/citations?user=bGwfkakAAAAJ)\*, and [Saehoon Kim](https://scholar.google.com/citations?user=_ZfueMIAAAAJ) at [Kakao Brain](https://www.kakaobrain.com).
(*: Equal contribution)

* This repository is an official implementation of the paper [Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity](https://arxiv.org/abs/2111.14330). 
* The code and some instructions are built upon the official [Deformable DETR repository](https://github.com/fundamentalvision/Deformable-DETR).



# Introduction

**TL; DR.** Sparse DETR is an efficient end-to-end object detector that **sparsifies encoder tokens** by using the learnable DAM(Decoder Attention Map) predictor. It achieves better performance than Deformable DETR even with only 10% encoder queries on the COCO dataset.

<p align="center">
<img src="./figs/dam_creation.png" height=350>
</p>

**Abstract.** DETR is the first end-to-end object detector using a transformer encoder-decoder architecture and demonstrates competitive performance but low computational efficiency on high resolution feature maps.
The subsequent work, Deformable DETR, enhances the efficiency of DETR by replacing dense attention with deformable attention, which achieves 10x faster convergence and improved performance. 
Deformable DETR uses the multiscale feature to ameliorate performance, however, the number of encoder tokens increases by 20x compared to DETR, and the computation cost of the encoder attention remains a bottleneck.
In our preliminary experiment, we observe that the detection performance hardly deteriorates even if only a part of the encoder token is updated.
Inspired by this observation, we propose *Sparse DETR* that selectively updates only the tokens expected to be referenced by the decoder, thus help the model effectively detect objects.
In addition, we show that applying an auxiliary detection loss on the selected tokens in the encoder improves the performance while minimizing computational overhead.
We validate that *Sparse DETR* achieves better performance than Deformable DETR even with only 10\% encoder tokens on the COCO dataset.
Albeit only the encoder tokens are sparsified, the total computation cost decreases by 38\% and the frames per second (FPS) increases by 42\% compared to Deformable DETR.


# Installation

## Requirements

We have tested the code on the following environments: 
* Python 3.7.7 / Pytorch 1.6.0 / torchvisoin 0.7.0 / CUDA 10.1 / Ubuntu 18.04
* Python 3.8.3 / Pytorch 1.7.1 / torchvisoin 0.8.2 / CUDA 11.1 / Ubuntu 18.04

Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```

## Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

# Usage

## Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as follows:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

## Training

### Training on a single node

For example, the command for training Sparse DETR with the keeping ratio of 10% on 8 GPUs is as follows:

```bash
$ GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/swint_sparse_detr_rho_0.1.sh
```

### Training on multiple nodes

For example, the command Sparse DETR with the keeping ratio of 10% on 2 nodes of each with 8 GPUs is as follows:

On node 1:

```bash
$ MASTER_ADDR=<IP address of node 1> NODE_RANK=0 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/swint_sparse_detr_rho_0.1.sh
```

On node 2:

```bash
$ MASTER_ADDR=<IP address of node 2> NODE_RANK=1 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/swint_sparse_detr_rho_0.1.sh
```

### Direct argument control

```bash
# Deformable DETR (with bounding-box-refinement and two-stage argument, if wanted)
$ GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 python main.py --with_box_refine --two_stage
# Efficient DETR (with the class-specific head as describe in their paper)
$ GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 python main.py --with_box_refine --two_stage --eff_query_init --eff_specific_head
# Sparse DETR (with the keeping ratio of 10% and encoder auxiliary loss)
$ GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 python main.py --with_box_refine --two_stage --eff_query_init --eff_specific_head --rho 0.1 --use_enc_aux_loss
```

### Some tips to speed-up training
* If your file system is slow to read images, you may consider enabling '--cache_mode' option to load the whole dataset into memory at the beginning of training.
* You may increase the batch size to maximize the GPU utilization, according to GPU memory of yours, e.g., set '--batch_size 3' or '--batch_size 4'.

## Evaluation

You can get the pre-trained model of Sparse DETR (the link is in "Main Results" session), then run the following command to evaluate it on COCO 2017 validation set:

```bash
# Note that you should run the command with the corresponding configuration.
$ ./configs/swint_sparse_detr_rho_0.1.sh --resume <path to pre-trained model> --eval
```

You can also run distributed evaluation by using ```./tools/run_dist_launch.sh```.

# Main Results
The tables below demonstrate the detection performance of Sparse DETR on the COCO 2017 validation set when using different backbones. 
* **Top-k** : sampling the top-k object queries instead of using the learned object queries(as in Efficient DETR).
* **BBR** : performing bounding box refinement in the decoder block(as in Deformable DETR).
* The **encoder auxiliary loss** proposed in our paper is only applied to Sparse DETR.
* **FLOPs** and **FPS** are measured in the same way as used in Deformable DETR. 
* Refer to **Table 1** in the paper for more details.



## ResNet-50 backbone
| Method             | Epochs | ρ   | Top-k & BBR | AP   | #Params(M) | GFLOPs | B4FPS | Download |
|:------------------:|:------:|:---:|:-----------:|:----:|:----------:|:------:|:-----:|:--------:|
| Faster R-CNN + FPN | 109    | N/A |             | 42.0 | 42M        | 180G   | 26    |          |
| DETR               | 50     | N/A |             | 35.0 | 41M        | 86G    | 28    |          |
| DETR               | 500    | N/A |             | 42.0 | 41M        | 86G    | 28    |          |
| DETR-DC5           | 500    | N/A |             | 43.3 | 41M        | 187G   | 12    |          |
| PnP-DETR           | 500    | 33% |             | 41.1 |            |        |       |          |
|                    | 500    | 50% |             | 41.8 |            |        |       |          |
| PnP-DETR-DC5       | 500    | 33% |             | 42.7 |            |        |       |          |
|                    | 500    | 50% |             | 43.1 |            |        |       |          |
| Deformable-DETR    | 50     | N/A |             | 43.9 | 39.8M      | 172.9G | 19.1  |          |
|                    | 50     | N/A | o           | 46.0 | 40.8M      | 177.3G | 18.2  |          |
| Sparse-DETR        | 50     | 10% | o           | 45.3 | 40.9M      | 105.4G | 26.5  | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_10.pth)     |
|                    | 50     | 20% | o           | 45.6 | 40.9M      | 112.9G | 24.8  | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_20.pth)     |
|                    | 50     | 30% | o           | 46.0 | 40.9M      | 120.5G | 23.2  | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_30.pth)     |
|                    | 50     | 40% | o           | 46.2 | 40.9M      | 128.0G | 21.8  | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_40.pth)     |
|                    | 50     | 50% | o           | 46.3 | 40.9M      | 135.6G | 20.5  | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_50.pth)     |



## Swin-T backbone
| Method          | Epochs | ρ   | Top-k & BBR | AP   | #Params(M) | GFLOPs | B4FPS | Download |
|:---------------:|:------:|:---:|:-----------:|:----:|:----------:|:------:|:-----:|:--------:|
| DETR            | 50     | N/A |             | 35.9 | 45.0M      | 91.6G  | 26.8  |          |
| DETR            | 500    | N/A |             | 45.4 | 45.0M      | 91.6G  | 26.8  |          |
| Deformable-DETR | 50     | N/A |             | 45.7 | 40.3M      | 180.4G | 15.9  |          |
|                 | 50     | N/A | o           | 48.0 | 41.3M      | 184.8G | 15.4  |          |
| Sparse-DETR     | 50     | 10% | o           | 48.2 | 41.4M      | 113.4G | 21.2  | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_swint_10.pth)     |
|                 | 50     | 20% | o           | 48.8 | 41.4M      | 121.0G | 20    | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_swint_20.pth)     |
|                 | 50     | 30% | o           | 49.1 | 41.4M      | 128.5G | 18.9  | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_swint_30.pth)     |
|                 | 50     | 40% | o           | 49.2 | 41.4M      | 136.1G | 18    | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_swint_40.pth)     |
|                 | 50     | 50% | o           | 49.3 | 41.4M      | 143.7G | 17.2  | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_swint_50.pth)     |


## Initializing ResNet-50 backbone with SCRL
The performance of Sparse DETR can be further improved when the backbone network is initialized with the `SCRL`([Spatially Consistent Representation Learning](https://arxiv.org/abs/2103.06122)) that aims to learn dense representations in a self-supervised way, compared to the default initialization with the ImageNet pre-trained one, denoted as `IN-sup` in the table below. 
* We obtained pre-trained weights from [Torchvision](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#sphx-glr-beginner-finetuning-torchvision-models-tutorial-py) for `IN-sup`, and the [SCRL GitHub repository](https://github.com/kakaobrain/scrl) for `SCRL`.
* To reproduce the `SCRL` results, add `--scrl_pretrained_path <downloaded_filepath>` to the training command.
 
| Method      | ρ   | AP(IN-sup) | AP(SCRL) | AP(gain) | Download |
|:-----------:|:---:|:-----------:|:--------:|:--------:|:--------:|
| Sparse DETR | 10% | 45.3        | 46.9     | +1.6     | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_scrl_10.pth)     |
|             | 20% | 45.6        | 47.2     | +1.7     | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_scrl_20.pth)     |
|             | 30% | 46.0        | 47.4     | +1.4     | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_scrl_30.pth)     |
|             | 40% | 46.2        | 47.7     | +1.5     | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_scrl_40.pth)     |
|             | 50% | 46.3        | 47.9     | +1.6     | [link](https://arena.kakaocdn.net/brainrepo/sparse_detr/sparse_detr_r50_scrl_50.pth)     |


# Citation
If you find Sparse DETR useful in your research, please consider citing:
```bibtex
@inproceedings{roh2022sparse,
  title={Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity},
  author={Roh, Byungseok and Shin, JaeWoong and Shin, Wuhyun and Kim, Saehoon},
  booktitle={ICLR},
  year={2022}
}
```

# License

This project is released under the [Apache 2.0 license](./LICENSE).
Copyright 2021 [Kakao Brain Corp](https://www.kakaobrain.com). All Rights Reserved.
