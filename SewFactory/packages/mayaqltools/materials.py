# from PIL import Image
import numpy as np
import os
import time
import json
import random

import colorsys
from array import array
import ctypes

# Maya
from maya import cmds
from maya import OpenMaya
import maya.api.OpenMaya as OM

# Arnold
# import mtoa.utils as mutils
# from mtoa.cmds.arnoldRender import arnoldRender
# import mtoa.core

from mayaqltools import utils
reload(utils)

class GarmentMaterials(object):
    """
        Describes the materials for rendering.
            mtl_scene: a scene file contains all the supportted mtls

        Note: support 4 kinds of mtls now (default, cotton, velvet, silk).
              Need input mtl resources first. 
        Assumes: the code is based on the input mtl resources. 

    """

    def __init__(self, mtl_scene_file, mtl_textures_path):
        self.mtl_scene_file = mtl_scene_file
        self.mtl_textures_path = mtl_textures_path
        self.material_types = ['cotton', 'silk', 'velvet', "default"]
        # self.materials = {}
        materials, material_namespace = self._load_maya_materials(self.mtl_scene_file, namespace="imported_mtl")
        self.materials, self.mtl_namespace = materials, material_namespace
        # load extra for two clothes (top and down)
        self.extr_materials, self.extr_namespace = self._load_maya_materials(self.mtl_scene_file, namespace="extra_mtl")

        # default materials (used for prediction)
        mt, colors = self._make_default_materials()
        self.materials["default"] = mt
        mt_extr, _ = self._make_default_materials(extr=True)
        self.extr_materials["default"] = mt_extr

        self.default_colors = colors 


    
    def random_load_materials(self, mtl_type, pattern_path="", color="", extr=False):
        assert mtl_type in self.material_types, "not support this mtl_type: {}".format(mtl_type)
        if mtl_type == "cotton":
            cot_mtl, cot_mtl_sg, random_params = self._random_cotton_mtls(pattern_path=pattern_path, extr=extr)
            return cot_mtl, cot_mtl_sg, random_params
        elif mtl_type == "silk":
            silk_mtl, silk_mtl_sg, random_params = self._random_silk_mtls(extr=extr)
            return silk_mtl, silk_mtl_sg, random_params
        elif mtl_type == "velvet":
            vel_mtl, vel_mtl_sg, random_params = self._random_velvet_mtls(pattern_path=pattern_path, extr=extr)
            return vel_mtl, vel_mtl_sg, random_params
        elif mtl_type == "default":
            default_shader, default_shader_sg = self._set_default_materials(color, extr=extr)
            return default_shader, default_shader_sg, None
        else:
            raise NotImplementedError("Not supported mtl type: {}".format(mtl_type))
    

    def restore_materials(self, mtl_type, pattern_path="", color="", restore_config=None, extr=False):
        if restore_config is None:
            print("Cannot restore, random")
            return self.random_load_materials(mtl_type, pattern_path, color, extr=extr)
        else:
            if mtl_type == "cotton":
                cot_mtl, cot_mtl_sg, random_params = self._restore_cotton_mtls(pattern_path=pattern_path, restore_config=restore_config, extr=extr)
                return cot_mtl, cot_mtl_sg, random_params
            elif mtl_type == "velvet":
                vel_mtl, vel_mtl_sg, random_params = self._restore_velvet_mtls(pattern_path=pattern_path, restore_config=restore_config, extr=extr)
                return vel_mtl, vel_mtl_sg, random_params
            elif mtl_type == "silk":
                silk_mtl, silk_mtl_sg, random_params = self._restore_silk_mtls(restore_config=restore_config, extr=extr)
                return silk_mtl, silk_mtl_sg, random_params
            elif mtl_type == 'default':
                default_mtl, default_mtl_sg = self._restore_default_materials(restore_config=restore_config, extr=extr)
                return default_mtl, default_mtl_sg, None
            else:
                raise NotImplementedError("Not supported mtl type: {}".format(mtl_type))


    
    def _make_default_materials(self, extr=False):
        default_shader = cmds.shadingNode('aiStandardSurface', asShader=True, name="default_shader" if not extr else "extr_default_shader")
        default_shader_sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=default_shader + '_sg')
        cmds.connectAttr(default_shader + ".outColor", default_shader_sg + ".surfaceShader")
        cmds.setAttr(default_shader + ".base", 1.0)
        cmds.setAttr(default_shader + ".specular", 0)
        cmds.setAttr(default_shader + ".sheen", 0.2)
        colors = {"bottoms": [0.722, 0.388, 0.267], "tops": [0.259, 0.525, 0.388], "one_pieces": [0.835, 0.557, 0.529],
                  "gt_tops": [float(213.0/255), float(217.0/255), float(177.0/255)], 
                  "gt_bottoms": [float(151.0/255), float(162.0/255), float(185.0/255)], 
                  "gt_one_pieces": [float(219.0/255), float(173.0/255), float(145.0/255)]}
        default_type = "one_pieces"
        cmds.setAttr(default_shader + '.baseColor', colors[default_type][0], colors[default_type][1], colors[default_type][2])
        return (default_shader, default_shader_sg), colors
    
    
    def _set_default_materials(self, color, extr=False):
        assert color in self.default_colors, "not support this color for default mtl: {}".format(color)
        default_shader, default_shader_sg = self.materials['default'] if not extr else self.extr_materials['default']
        cmds.setAttr(default_shader + '.baseColor', self.default_colors[color][0], self.default_colors[color][1], self.default_colors[color][2])
        return default_shader, default_shader_sg
    
    def _restore_default_materials(self, restore_config, extr=False):
        default_shader, default_shader_sg = self.materials['default'] if not extr else self.extr_materials['default']
        restore_config = [float(restore_config[0] / 255.0) if restore_config[0] > 1 else restore_config[0],
                          float(restore_config[1] / 255.0) if restore_config[1] > 1 else restore_config[1],
                          float(restore_config[2] / 255.0) if restore_config[2] > 1 else restore_config[2]]
        cmds.setAttr(default_shader + '.baseColor', restore_config[0], restore_config[1], restore_config[2])
        return default_shader, default_shader_sg


    def _load_maya_materials(self, mtl_scene_file, namespace="imported_mtl"):
        # mtl file: actually it is a maya scene file, but only contains materials
        print("MTL file: {}, namespace: {}".format(mtl_scene_file, namespace))
        self.mtl_scene_file = mtl_scene_file

        before = set(cmds.ls())
        cmds.file(mtl_scene_file, i=True, namespace=namespace)
        new_objects = set(cmds.ls()) - before

        # Maya may modify namespace for uniquness
        mtl_namespace = new_objects.pop().split(':')[0] + '::' 

        # load all garment shaders
        garment_mtls = cmds.ls(mtl_namespace + '*garment*', materials=True)
        
        materials = {}
        for mtl in garment_mtls:
            mtl_sg = cmds.ls('*{}_sg*'.format(mtl))[0]
            for mtl_type in self.material_types:
                if mtl_type in mtl:
                    materials[mtl_type] = (mtl, mtl_sg)
        return materials, mtl_namespace
    
    def _restore_silk_mtls(self, restore_config=None, extr=False):
        silk_mtl, silk_mtl_sg = self.materials["silk"] if not extr else self.extr_materials["silk"]
        connections = cmds.listConnections(silk_mtl)
        rgb_layer = [connect for connect in connections if "silk" in connect and "rgbalayer" in connect][0]
        cmds.setAttr(rgb_layer + '.input1', restore_config["rgb_layer_color1"][0], restore_config["rgb_layer_color1"][1], restore_config["rgb_layer_color1"][2])
        cmds.setAttr(rgb_layer + '.input2', restore_config["rgb_layer_color2"][0], restore_config["rgb_layer_color2"][1], restore_config["rgb_layer_color2"][2])
        return silk_mtl, silk_mtl_sg, None

    def _random_silk_mtls(self, extr=False):
        random_params = {}
        silk_mtl, silk_mtl_sg = self.materials["silk"] if not extr else self.extr_materials["silk"]
        connections = cmds.listConnections(silk_mtl)
        rgb_layer = [connect for connect in connections if "silk" in connect and "rgbalayer" in connect][0]
        # random color 1
        hsv = (random.random(), random.uniform(0, 1), random.uniform(0, 1))
        rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
        cmds.setAttr(rgb_layer + '.input1', rgb[0], rgb[1], rgb[2])

        random_params["rgb_layer_color1"] = rgb 
        # random color 2
        hsv2 = (hsv[0], random.uniform(hsv[1], 1), random.uniform(hsv[2], 1))
        rgb2 = colorsys.hsv_to_rgb(hsv2[0], hsv2[1], hsv2[2])
        cmds.setAttr(rgb_layer + '.input2', rgb2[0], rgb2[1], rgb2[2])

        random_params["rgb_layer_color2"] = rgb 
        return silk_mtl, silk_mtl_sg, random_params
    


    def _restore_cotton_mtls(self, pattern_path="", restore_config=None, extr=False):
        cot_mtl, cot_mtl_sg = self.materials["cotton"] if not extr else self.extr_materials["cotton"]
        connects = cmds.listConnections(cot_mtl)
        # restore bump depth
        cot_bump = [connect for connect in connects if 'cotton' in connect and 'bump' in connect][0]
        cmds.setAttr(cot_bump + '.bumpDepth', restore_config["bumpDepth"])

        # restore pattern file
        cot_rgbalayer = [connect for connect in connects if 'cotton' in connect and 'rgbalayer' in connect][0]
        pattern_file = restore_config["fileTextureName"]
        if not os.path.exists(pattern_file):
            pattern_file = os.path.join(pattern_path, os.path.basename(pattern_file))
        cot_pattern_file = [connect for connect in cmds.listConnections(cot_rgbalayer) if "cotton" in connect and "pattern_file" in connect][0]
        cmds.setAttr(cot_pattern_file + '.fileTextureName', pattern_file, type='string')

        # restore rgb color
        cmds.setAttr(cot_rgbalayer + '.input2',  restore_config["rgbalayer_color"][0], restore_config["rgbalayer_color"][1], restore_config["rgbalayer_color"][2])
        connects = cmds.listConnections(cot_pattern_file)
        return cot_mtl, cot_mtl_sg, None

    def _random_cotton_mtls(self, pattern_path="", extr=False):
        tex_path = pattern_path if os.path.exists(pattern_path) else self.mtl_textures_path
        random_params = {}
        cot_mtl, cot_mtl_sg = self.materials["cotton"] if not extr else self.extr_materials["cotton"]
        connects = cmds.listConnections(cot_mtl)

        # random bump depth
        cot_bump = [connect for connect in connects if 'cotton' in connect and 'bump' in connect][0]
        random_bump_depth = random.uniform(0, 0.1)
        cmds.setAttr(cot_bump + '.bumpDepth', random_bump_depth)
        random_params["bumpDepth"] = random_bump_depth

        # random pattern file
        cot_rgbalayer = [connect for connect in connects if 'cotton' in connect and 'rgbalayer' in connect][0]
        cot_pattern_file = [connect for connect in cmds.listConnections(cot_rgbalayer) if "cotton" in connect and "pattern_file" in connect][0]
        if os.path.exists(tex_path):
            if os.path.isfile(tex_path):
                random_pattern_file = tex_path
                random_params["fileTextureName"] = tex_path
                cmds.setAttr(cot_pattern_file + '.fileTextureName', random_pattern_file, type='string')
            else:
                pattern_files = [fn for fn in os.listdir(tex_path) if os.path.splitext(fn)[-1] in ['.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG']]
                print("!!! Total texture_fns: {}".format(len(pattern_files)))
                if len(pattern_files) > 0:
                    random_pattern_file = os.path.join(tex_path, random.choice(pattern_files))
                    print("---> Current random texture fn: {}".format(random_pattern_file))
                    cmds.setAttr(cot_pattern_file + '.fileTextureName', random_pattern_file, type='string')

                    random_params["fileTextureName"] = random_pattern_file
        
        # random rbgalayer color
        pattern_file_name = cmds.getAttr(cot_pattern_file + '.fileTextureName')
        pattern_image = OM.MImage()
        pattern_image.readFromFile(pattern_file_name)
        pattern_image.resize(1, 1)
        data_ptr = ctypes.cast(pattern_image.pixels(), ctypes.POINTER(ctypes.c_char))
        dominate_color = array('B', ctypes.string_at(data_ptr, 4))
        r, g, b = dominate_color[:3]
        hsv = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        rhsv = (hsv[0], random.uniform(0, 0.5) if hsv[1] > 0.5 else hsv[1], random.uniform(0, 0.5) if hsv[2] > 0.5 else hsv[2]) 
        rrgb = colorsys.hsv_to_rgb(rhsv[0], rhsv[1], rhsv[2])
        cmds.setAttr(cot_rgbalayer + '.input2', rrgb[0], rrgb[1], rrgb[2])
        random_params["rgbalayer_color"] = rrgb

        connects = cmds.listConnections(cot_pattern_file)
        cot_uv_placed_tex = [connect for connect in connects if "cotton" in connect and "uv_place2d_texture" in connect][0]
        cmds.setAttr(cot_uv_placed_tex + '.repeatUV', 1, 1)

        return cot_mtl, cot_mtl_sg, random_params

    def _restore_velvet_mtls(self, pattern_path="", restore_config=None, extr=False):
        # import pdb; pdb.set_trace()
        vel_mtl, vel_mtl_sg = self.materials["velvet"] if not extr else self.extr_materials['velvet']
        vel_mtl1, vel_mtl2 = [connect for connect in sorted(cmds.listConnections(vel_mtl)) if "velvet_layer" in connect][:2]
        cmds.setAttr(vel_mtl1 + '.baseColor', restore_config["vel_mtl1_base_color"][0], restore_config["vel_mtl1_base_color"][1], restore_config["vel_mtl1_base_color"][2])
        cmds.setAttr(vel_mtl2 + '.baseColor', restore_config["vel_mtl2_base_color"][0], restore_config["vel_mtl2_base_color"][1], restore_config["vel_mtl2_base_color"][2])
        vel_pattern_file = [connect for connect in cmds.listConnections(vel_mtl) if 'velvet' in connect and 'pattern' in connect][0]
        pattern_file = restore_config["fileTextureName"]
        if not os.path.exists(pattern_file):
            pattern_file = os.path.join(pattern_path, os.path.basename(pattern_file))
        cmds.setAttr(vel_pattern_file + '.fileTextureName', pattern_file, type='string')
        return vel_mtl, vel_mtl_sg, None

    def _random_velvet_mtls(self, pattern_path="", extr=False):
        tex_path = pattern_path if os.path.exists(pattern_path) else self.mtl_textures_path
        random_params ={}

        vel_mtl, vel_mtl_sg = self.materials["velvet"] if not extr else self.extr_materials['velvet']
        vel_mtl1, vel_mtl2 = [connect for connect in sorted(cmds.listConnections(vel_mtl)) if "velvet_layer" in connect][:2]
        # random color 1
        rgb = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        cmds.setAttr(vel_mtl1 + '.baseColor', rgb[0], rgb[1], rgb[2])

        random_params["vel_mtl1_base_color"] = rgb 

        # random color 2
        rgb = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        cmds.setAttr(vel_mtl2 + '.baseColor', rgb[0], rgb[1], rgb[2])

        random_params["vel_mtl2_base_color"] = rgb 

        # random pattern file
        vel_pattern_file = [connect for connect in cmds.listConnections(vel_mtl) if 'velvet' in connect and 'pattern' in connect][0]
        if os.path.exists(tex_path):
            pattern_files = [fn for fn in os.listdir(tex_path) if os.path.splitext(fn)[-1] in ['.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG']]
            print("!!! Total texture_fns: {}".format(len(pattern_files)))
            if len(pattern_files) > 0:
                random_pattern_file = os.path.join(tex_path, random.choice(pattern_files))
                print("---> Current random texture fn: {}".format(random_pattern_file))
                cmds.setAttr(vel_pattern_file + '.fileTextureName', random_pattern_file, type='string')

                random_params["fileTextureName"] = random_pattern_file
        
        connects = cmds.listConnections(vel_pattern_file)
        vel_uv_placed_tex = [connect for connect in connects if "velvet" in connect and "uv_place2d_texture" in connect][0]
        
        cmds.setAttr(vel_uv_placed_tex + '.repeatUV', 1, 1)
        return vel_mtl, vel_mtl_sg, random_params
        
