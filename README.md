# PRODEN

This is the code for the paper: Progressive Identification of True Labels for Partial-Label Learning

Jiaqi Lv, Miao Xu, Lei Feng, Gang Niu, Xin Geng, Masashi Sugiyama

To be presented at ICML 2020.

## Setups

All code was developed and tested on a single machine equiped with a NVIDIA Tesla V100 GPU. The environment is as bellow:
- Python 3.6.8
- Numpy 1.16.4
- Cuda 10.1.168

## Quick Start

Here is an example:
```
python main.py --dataset mnist --model linear --partial_type binomial --partial_rate 0.1
```

## Results

The test results and transductive accuracy are printed in result/ by default.

Contact: Jiaqi Lv (lvjiaqi@seu.edu.cn).
