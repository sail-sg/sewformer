# Sewformer
This is the official implementation of [Towards Garment Sewing Pattern Reconstruction from a Single Image](https://arxiv.org/abs/2311.04218v1).

[Lijuan Liu](https://scholar.google.com/citations?user=nANxp5wAAAAJ&hl=en)<sup> *</sup>,
[Xiangyu Xu](https://xuxy09.github.io/)<sup> *</sup>,
[Zhijie Lin](https://scholar.google.com/citations?user=xXMj6_EAAAAJ&hl=zh-CN)<sup> *</sup>,
[Jiabing Liang]()<sup> *</sup>,
[Shuicheng Yan](https://yanshuicheng.info/)<sup>&dagger;<sup></sup>,  
ACM Transactions on Graphics (SIGGRAPH Asia 2023)

### [Project](https://sewformer.github.io/) | [Paper](https://arxiv.org/abs/2311.04218v1)

<img src="SewFactory/assets/representative.jpg">

---------------------------
## How to Generate SewFactory Dataset

#### Requirement 
$\color{#FF0000}{todo:}$ 需要的操作系统，硬件等

#### Installation

* Dependencies: Autodesk Maya 2020 with Arnold, [Qualoth](http://www.fxgear.net/vfx-software?locale=en) 

$\color{#FF0000}{todo:}$ 是否需要一个 link to Arnold?

* Maya Python Environment: [Numpy](https://forums.autodesk.com/t5/maya-programming/guide-how-to-install-numpy-scipy-in-maya-windows-64-bit/td-p/5796722)
* Python Environement: `python3.9 -m pip install numpy scipy svglib svgwrite psutil wmi`

$\color{#FF0000}{todo:}$ 上面两条都保含了怎么装numpy和scipy，应该选哪一个？

#### Configuration

1. `cd ./DatasetGenerator` and add `./packages` to `PYTHONPATH` on your system for correct importing of our custom modules.

$\color{#FF0000}{todo:}$ DatasetGenerator和packages这两个文件夹在哪里？please make sure all the paths can be followed

2. Download SMPL related files to `.\meta_infos\fbx_metas\`

$\color{#FF0000}{todo:}$ SMPL related files包含哪些？去哪里下载？



#### Generation
1. Sample sewing pattern designs from the template.
`python3.9 .\data_generator\pattern_gen.py`
2. Generate posed smpl files with different pose sequences.
`& 'C:\Program Files\Autodesk\Maya2020\bin\mayapy.exe' .\data_generator\fbx_anime.py`


3. Simulate the garmetns with different poses.
`& 'C:\Program Files\Autodesk\Maya2020\bin\mayapy.exe' .\data_generator\data_sim.py`

## How to Run SewFormer

### Installation
* `conda env create -f environment.yaml`

### Test
1. Configure environment. `conda activate garment`

$\color{#FF0000}{todo:}$ download pretrained checkpoint?

2. `cd ./SewFormer` 

    for in-the-wild images:
    `python inference.py -c configs/test.yaml -d assets/data/real_images -t real -o outputs/real` 

    for deepfashion dataset:
    `python inference.py -c configs/test.yaml -d assets/data/deepfashion -t deepfashion -o outputs/deepfashion` 

    $\color{#FF0000}{todo:}$ for SewFactory dataset:

3. `cd path_to_garment_dev/SewFactory` and run `& 'C:\Program Files\Autodesk\Maya2020\bin\mayapy.exe' .\data_generator\deepfashion_sim.py` to simulate the predicted sew patterns. (Please prepare the SMPL prediction results with [RSC-Net](https://github.com/xuxy09/RSC-Net) and update the predicted data root specified in `deepfashion_sim.py`.)

$\color{#FF0000}{todo:}$ path_to_garment_dev是指哪里？

### Train


### Acknowledgement
- [Dataset of 3D Garments with Sewing patterns](https://github.com/maria-korosteleva/Garment-Pattern-Generator/tree/master)


### BibTex
Please cite this paper if you find the code/model helpful in your research:
```
 @article{liu2023sewformer,
    author      = {Liu, Lijuan and Xu, Xiangyu and Lin, Zhijie and Liang, Jiabin and Yan, Shuicheng},
    title       = {Towards Garment Sewing Pattern Reconstruction from a Single Image},
    journal     = {ACM Transactions on Graphics (SIGGRAPH Asia)},
    year        = {2023}
  }
```


