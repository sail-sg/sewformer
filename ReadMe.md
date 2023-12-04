# Sewformer
This is the official implementation of **Towards Garment Sewing Pattern Reconstruction from a Single Image**.

[Lijuan Liu](https://scholar.google.com/citations?user=nANxp5wAAAAJ&hl=en)<sup> *</sup>,
[Xiangyu Xu](https://xuxy09.github.io/)<sup> *</sup>,
[Zhijie Lin](https://scholar.google.com/citations?user=xXMj6_EAAAAJ&hl=zh-CN)<sup> *</sup>,
[Jiabing Liang]()<sup> *</sup>,
[Shuicheng Yan](https://yanshuicheng.info/)<sup>&dagger;<sup></sup>,  
ACM Transactions on Graphics (SIGGRAPH Asia 2023)

### [Project page](https://sewformer.github.io/) | [Paper](https://arxiv.org/abs/2311.04218v1)

<img src="SewFactory/assets/representative.jpg">

---------------------------
## SewFactory Generation

#### Installation

* Dependencies: Autodesk Maya 2020 with Arnold, [Qualoth](http://www.fxgear.net/vfx-software?locale=en)

* Maya Python Environment: [Numpy](https://forums.autodesk.com/t5/maya-programming/guide-how-to-install-numpy-scipy-in-maya-windows-64-bit/td-p/5796722)
* Python Environement: `python3.9 -m pip install numpy scipy svglib svgwrite psutil wmi`

#### Configuration

1. `cd ./DatasetGenerator` and add `./packages` to `PYTHONPATH` on your system for correct importing of our custom modules.
2. Download SMPL related files to `.\meta_infos\fbx_metas\`
#### Generation
1. Sample sewing pattern designs from the template.
`python3.9 .\data_generator\pattern_gen.py`
2. Generate posed smpl files with different pose sequences.
`& 'C:\Program Files\Autodesk\Maya2020\bin\mayapy.exe' .\data_generator\fbx_anime.py`
3. Simulate the garmetns with different poses.
`& 'C:\Program Files\Autodesk\Maya2020\bin\mayapy.exe' .\data_generator\data_sim.py`

## SewFormer Prediction

#### Installation
* `conda env create -f environment.yaml`

### Prediction
1. Configure environment. `conda activate garment`
2. `cd path_to_garment_dev` and run

    `python inference.py -c configs/test.yaml -d assets/data/real_images -t real -o outputs/real` for real images

    `python inference.py -c configs/test.yaml -d assets/data/deepfashion -t deepfashion -o outputs/deepfashion` for deepfashion dataset
3. `cd path_to_garment_dev/SewFactory` and run `& 'C:\Program Files\Autodesk\Maya2020\bin\mayapy.exe' .\data_generator\deepfashion_sim.py` to simulate the predicted sew patterns. (Please prepare the SMPL prediction results and update the predicted data root specified in `deepfashion_sim.py`.)


### Acknowledgement
- [Dataset of 3D Garments with Sewing patterns](https://github.com/maria-korosteleva/Garment-Pattern-Generator/tree/master)


### BibTex
Please cite this paper if you find the code/model helpful in your research:
```
 @article{liu2023sewformer,
    author      = {Liu, Lijuan and Xu, Xiangyu and Lin, Zhijie and Liang, Jiabin and Yan, Shuicheng},
    title       = {Towards Garment Sewing Pattern Reconstruction from a Single Image},
    journal   = {ACM Transactions on Graphics (SIGGRAPH Asia)},
    year        = {2023}
  }
```


