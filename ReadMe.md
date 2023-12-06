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

#### Installation (Windows, GPU Enabled)
1. Dependencies for Maya and related python environment:
 * Dependencies: Autodesk Maya 2020 with two plugins: Arnold and [Qualoth](http://www.fxgear.net/vfx-software?locale=en) 
    > We have only tested the sewfactory on this version of dependencies, and using the latest version may result in errors.
    > Please prepare Arnold license to render images without watermarks.

  * Maya Python Environment: [Numpy for python2.7](https://forums.autodesk.com/t5/maya-programming/guide-how-to-install-numpy-scipy-in-maya-windows-64-bit/td-p/5796722)

2. Python environment:
  * `python3.9 -m pip install numpy scipy svglib svgwrite psutil wmi`

#### Configuration

1. Clone the source code to `path_to_dev` and `cd path_to_dev/SewFactory`, add `path_to_dev/SewFactory/packages` to `PYTHONPATH` on your system for correct importing of our custom modules.

2. Download [SMPLH Fbx model (SMPLH_female_010_207.fbx)](https://smpl.is.tue.mpg.de/) to `meta_infos\\fbx_metas`.
3. Prepare human skin textures to `examples\\skin_textures` (for example,  [SURREAL](lsh.paris.inria.fr/SURREAL/smpl_data/textures.tar.gz)).
4. Prepare human poses to `examples\\human_poses` (for example, [AMASS](https://amass.is.tue.mpg.de/))
5. Collect some images used for garment textures and put them into `examples\\garment_textures`

#### Generation
1. Sample sewing pattern designs from the template (Please update the parameters in `meta_infos/configs/dataset_config.yaml` to generate your dataset. The generated sewing pattern is located at `examples`).
`python3.9 .\data_generator\pattern_gen.py -c meta_infos/configs/dataset_config.yaml -o examples`
2. Generate posed smpl files with different pose sequences.
`path_to_maya2020\bin\mayapy.exe .\data_generator\fbx_anime.py -o examples\\posed_fbxs`

3. Simulate the garmetns with different poses (Please update the config file located in `meta_infos\\configs\\data_sim_configs.json` to make sure all the prepared data resource located correctly).
`path_to_maya2020\bin\mayapy.exe .\data_generator\data_sim.py`

## How to Run SewFormer

#### Installation (Ubuntu) and Configuration
* We provide an conda env yaml file and the runtime environment can be initialized with`conda env create -f environment.yaml`
* `cd path_to_dev/SewFormer` and download the pre-trained [checkpoint]((https://huggingface.co/liulj/garment)) and put it into `assets/ckpts`, then activate the environment `conda activate garment`. 

#### Training
* Download our provided dataset or generate your dataset with the provided code and put it into `path_to_sewfactory`, update the local paths in `system.json` to make sure the dataset setup correctly. Train the model with
`torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py -c configs/train.yaml`
The test results will located at the `output` in `system.json`.

#### Testing

1. Inference sewing patterns with the trained model or our pretrained model:
* for in-the-wild images:
    `python inference.py -c configs/test.yaml -d assets/data/real_images -t real -o outputs/real` 

* for deepfashion dataset:
    `python inference.py -c configs/test.yaml -d assets/data/deepfashion -t deepfashion -o outputs/deepfashion` 

3. Simulate the predicted results (back to Windows):
`cd path_to_dev/SewFactory` and run `path_to_maya\bin\mayapy.exe .\data_generator\deepfashion_sim.py` to simulate the predicted sew patterns. (Please prepare the SMPL prediction results with [RSC-Net](https://github.com/xuxy09/RSC-Net) and update the predicted data root specified in `deepfashion_sim.py`.)


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


