# Neural-Pose-Transfer

This is an implementation of the CVPR'20 paper "Neural Pose Transfer by Spatially Adaptive Instance Normalization".

Please check our [paper](https://arxiv.org/abs/2003.07254) and the [project webpage](https://jiashunwang.github.io/Neural-Pose-Transfer/) for more details.

#### Citation

If you use our code or paper, please consider citing:
```
@inproceedings{wang2020neural,
  title={Neural Pose Transfer by Spatially Adaptive Instance Normalization},
  author={Wang, Jiashun and Wen, Chao and Fu, Yanwei and Lin, Haitao and Zou, Tianyun and Xue, Xiangyang and Zhang, Yinda},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5831--5839},
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
We provide our code data_generation.py based on https://github.com/CalciferZh/SMPL and for more information about SMPL, please check https://smpl.is.tue.mpg.de/. The meshes with clothes are from https://virtualhumans.mpi-inf.mpg.de/mgn/.

## Data and Pre-trained model
We provide dataset and pre-trained model used in our paper, please download data from [data link](http://www.sdspeople.fudan.edu.cn/fuyanwei/download/NeuralPoseTransfer/data/), and download model weights from [model link](http://www.sdspeople.fudan.edu.cn/fuyanwei/download/NeuralPoseTransfer/ckpt/). The test data file lists are also provided, the mesh file order in file lists are `identiy pose gt`.
(Backup links: [Google Drive](https://drive.google.com/drive/folders/1ZduWjWn5sqbiU7aG2VSFm5YcdGudFTwk?usp=sharing))

## Running the demo
We provide the pre-trained model for the original method and maxpooling method and also two meshes for test. For you own data, please train the model by yourself, because the pose parameter space may be different. For human meshes with clothes, we recommend the max-pooling method.
```
python demo.py
```


## Training
We provide both original and max-pooling methods. (Max-pooling one is recommended). The original method has slightly better quantitative results. The max-pooling method is more convenient when dealing with identity and pose meshes with different number of vertices and this method produces smoother results.
```
python train.py
```

## Evaluation
evaluate.py is the code for evaluation.

## Acknowledgement
Part of our code is based on [SPADE](https://github.com/NVlabs/SPADE)，[3D-CODED](https://github.com/ThibaultGROUEIX/3D-CODED) and [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch
). Many thanks!

This work was supported in part by NSFC Projects (U1611461), Science and Technology Commission of Shanghai Municipality Projects (19511120700, 19ZR1471800), Shanghai Municipal Science and Technology Major Project (2018SHZDZX01), and Shanghai Research and Innovation Functional Program (17DZ2260900).

## License
Apache-2.0 License
