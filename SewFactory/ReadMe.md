## How to Generate SewFactory Dataset

### Installation (Windows, GPU Enabled)
1. Maya python environment:
 * Dependencies: Autodesk Maya 2020 with two plugins: Arnold and [Qualoth](http://www.fxgear.net/vfx-software?locale=en) 
    > We have only tested the sewfactory on this version of dependencies, and using the latest version may result in errors.
    > Please prepare Arnold license to render images without watermarks.

  * Maya Python Environment: [Numpy for python2.7](https://forums.autodesk.com/t5/maya-programming/guide-how-to-install-numpy-scipy-in-maya-windows-64-bit/td-p/5796722)

2. Windows python environment:
  * `python3.9 -m pip install numpy scipy svglib svgwrite psutil wmi`

### Configuration

1. `cd path_to_dev/SewFactory` and add `path_to_dev/SewFactory/packages` to `PYTHONPATH` on your system for correct importing of our custom modules.

2. Download [SMPLH Fbx model (SMPLH_female_010_207.fbx)](https://smpl.is.tue.mpg.de/) to `meta_infos\\fbx_metas`.
3. (Optional) Prepare human skin textures to `examples\\skin_textures` (for example,  [SURREAL](lsh.paris.inria.fr/SURREAL/smpl_data/textures.tar.gz)).
4. Prepare human poses to `examples\\human_poses` (for example, [AMASS](https://amass.is.tue.mpg.de/))
5. Collect some images used for garment textures and put them into `examples\\garment_textures`

### Generation
1. Sample sewing pattern designs from the template (Please update the parameters in `meta_infos\\configs\\dataset_config.yaml` to generate your dataset. The generated sewing patterns are located at `examples`).
`python3.9 .\data_generator\pattern_gen.py -c meta_infos/configs/dataset_config.yaml -o examples`
2. Generate posed smpl files with different pose sequences.
`path_to_maya2020\bin\mayapy.exe .\data_generator\fbx_anime.py -o examples\\posed_fbxs`

3. Simulate the garmetns with different poses (Please update the config file located in `meta_infos\\configs\\data_sim_configs.json` to ensure all the prepared data resource are correctly located).
`path_to_maya2020\bin\mayapy.exe .\data_generator\data_sim.py`

### Acknowledgement
- [Dataset of 3D Garments with Sewing patterns](https://github.com/maria-korosteleva/Garment-Pattern-Generator/tree/master)