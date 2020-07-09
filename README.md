# Neural-Pose-Transfer

This is an implementation of the CVPR'20 paper "Neural Pose Transfer by Spatially Adaptive Instance Normalization".

Please check our [paper](https://arxiv.org/abs/2003.07254) and the [project webpage](https://jiashunwang.github.io/Neural-Pose-Transfer/) for more details.

#### Citation

If you use this code for any purpose, please consider citing:
```
@inProceedings{wang2020npt,
  title={Neural Pose Transfer by Spatially Adaptive Instance Normalization},
  author={Jiashun Wang and Chao Wen and Yanwei Fu and Haitao Lin and Tianyun Zou and Xiangyang Xue and Yinda Zhang},
  booktitle={CVPR},
  year={2020}
}
```

## Dependencies

Requirements:
- python3.6
- numpy
- pytorch==1.1.0
- [pymesh](https://pymesh.readthedocs.io/en/latest/)

Our code has been tested with Python 3.6, Pytorch1.1.0, CUDA 9.0 on Ubuntu 16.04.

## Generating the data
We provide our code data_generation.py based on https://github.com/CalciferZh/SMPL and for more information about SMPL, please check https://smpl.is.tue.mpg.de/.

## Running the demo
We provide the pretrained model for the original method and also two meshes for test. For you own data, please train the model by yourself, because the pose parameter space may be different. For human meshes with clothes, we recommend the max-pooling method.
```
python demo.py
```


## Training
We provide both original and max-pooling methods. The original method has slightly better quantitative results. The max-pooling method is more convenient when dealing with identity and pose meshes with different number of vertices and this method produces smoother results.
```
python train.py
```

## Acknowledgement
Part of our code is based on [SPADE](https://github.com/NVlabs/SPADE)，[3D-CODED](https://github.com/ThibaultGROUEIX/3D-CODED) and [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch
). Many thanks!

This work was supported in part by NSFC Projects (U1611461), Science and Technology Commission of Shanghai Municipality Projects (19511120700, 19ZR1471800), Shanghai Municipal Science and Technology Major Project (2018SHZDZX01), and Shanghai Research and Innovation Functional Program (17DZ2260900).

## License
Apache-2.0 License
