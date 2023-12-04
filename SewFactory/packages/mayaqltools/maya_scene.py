# Basic
from __future__ import print_function
from __future__ import division
#from curses.ascii import BEL
from functools import partial
import copy
import errno
# from operator import sub
from random import random, uniform
from re import L
# from tkinter.tix import Tree
# from PIL import Image
import numpy as np
import os
import time
import json
import random

import colorsys
from array import array
import ctypes
import math

# Maya
from maya import cmds
from maya import OpenMaya
import maya.api.OpenMaya as OM

# Arnold
import mtoa.utils as mutils
from mtoa.cmds.arnoldRender import arnoldRender
import mtoa.core

from mayaqltools import utils
reload(utils)
# from mayaqltools import simple_materials
from mayaqltools import materials

class MayaScene(object):
    """
        Decribes scene setup that includes:
            # Mtl(s) & light(s): preload, donot move
            * floor & camera(s): preload, ajdust according to the body
        Assumes 
            * body the scene revolved aroung faces z+ direction
    """

    def __init__(self, props, clean_on_die=False):

        self.self_clean = clean_on_die
        self.props = props
        self.config = props['config'] if self.props is not None else None
        
        self._init_arnold()
        self._default_scene_setup()
        self.Mtls = materials.GarmentMaterials(self.config["materials"], self.config["textures_path"])


    def _init_arnold(self, ):
        """Ensure Arnold objects are launched in Maya & init GPU rendering settings"""

        objects = cmds.ls('defaultArnoldDriver')
        # print(objects)
        if not objects:  # Arnold objects not found
            # https://arnoldsupport.com/2015/12/09/mtoa-creating-the-defaultarnold-nodes-in-scripting/
            print('Initialized Arnold')
            mtoa.core.createOptions()
        # try:
        #     renderDevice = cmds.getAttr('defaultArnoldRenderOptions.renderDevice')
        # except:
        #     renderDevice = 0
        # print(renderDevice)
        cpu_render = False
        if cpu_render:
            cmds.setAttr('defaultArnoldRenderOptions.renderDevice', 0)  # turn on CPU rendering
            cmds.setAttr('defaultArnoldRenderOptions.AASamples', 5)  # 232 look good enough
            cmds.setAttr('defaultArnoldRenderOptions.GIDiffuseSamples',10)
            cmds.setAttr('defaultArnoldRenderOptions.GISpecularSamples',20)
        else:
            cmds.setAttr('defaultArnoldRenderOptions.renderDevice', 1)  # turn on GPPU rendering
            cmds.setAttr('defaultArnoldRenderOptions.AASamples', 10)  # increase sampling for clean results 

    # def load_materials(self, mtl_path=""):
    #     if not os.path.exists(mtl_path):
    #         print("mtl_path: {} not valid, use config instead".format(mtl_path))
    #         if self.config is None:
    #             print("Config is NONE, cannot import mtls")
    #             self.Mtls = None
    #         else:
    #             mtl_path = self.config["materials"]
    #     self.config['materials'] = mtl_path
    #     self.Mtls = simple_materials.SimpleMtls(mtl_path=mtl_path)
    

    def _default_scene_setup(self):
        # add floor
        floor_color = [0.8, 0.8, 0.8]
        floor, floor_shader, floor_sg = self._add_floor(floor_color)
        self.scene = {}
        self.scene["floor"], self.scene["floor_shader"], self.scene["floor_sg"] = floor, floor_shader, floor_sg

        # add light
        self.scene['light'] = mutils.createLocator('aiSkyDomeLight', asLight=True)
        # physicalSkyShader = cmds.createNode('aiPhysicalSky')
        # cmds.connectAttr(physicalSkyShader + ".outColor", self.scene['light'][0] + ".color")
        # self.scene['lightshader'] = physicalSkyShader

        # add camera
        # TODO
        cam_rotations = {"front": [0, 90, 0],
                         "left": [0, 180, 0],
                         "back": [0, -90, 0],
                         "right": [0, 0, 0],
                         "30_0": [0, 30, 0],
                         "60_0": [0, 60, 0],
                         "120_0": [0, 120, 0],
                         "150_0": [0, 150, 0],
                         "210_0": [0, 210, 0],
                         "240_0": [0, 240, 0],
                         "300_0": [0, 300, 0],
                         "330_0": [0, 330, 0],
                         "90_30": [-30, 90, 0],
                         "180_30": [-30, 180, 0],
                         "270_30": [-30, 270, 0],
                         "0_30": [-30, 0, 0],
                         "30_30": [-30, 30, 0],
                         "60_30": [-30, 60, 0],
                         "120_30": [-30, 120, 0],
                         "150_30": [-30, 150, 0],
                         "210_30": [-30, 210, 0],
                         "240_30": [-30, 240, 0],
                         "300_30": [-30, 300, 0],
                         "330_30": [-30, 330, 0],}

        self.scene["cameras"] = {}
        self.scene["camera_trans"] = {}
        for name, rot in cam_rotations.items():
            self.scene["cameras"][name] = self._add_default_camera(name, rotation=rot)

    def _add_floor(self, floor_color, width=1000, height=1000):
        floor = cmds.polyPlane(n='floor', w=width, h=height)
        # Make the floor non-renderable
        shape = cmds.listRelatives(floor[0], shapes=True)
        cmds.setAttr(shape[0] + '.primaryVisibility', 0)
        floor_shader, floor_sg = self._new_lambert(floor_color, floor[0])
        return floor[0], floor_shader, floor_sg
    
    def _new_lambert(self, color, target=None):
        """created a new shader node with given color"""
        shader = cmds.shadingNode('lambert', asShader=True)
        cmds.setAttr((shader + '.color'), color[0], color[1], color[2], type='double3')

        shader_group = self._create_shader_group(shader, target)
        if target is not None:
            cmds.sets(target, forceElement=shader_group)

        return shader, shader_group
    
    def _create_shader_group(self, material, name='shader'):
        """Create a shader group set for a given material (to be used in cmds.sets())"""
        shader_group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=name)
        cmds.connectAttr(material + '.outColor', shader_group + '.surfaceShader')
        return shader_group
    
    def _add_default_camera(self, name, rotation=[0, 0, 0], obj=None):
        """Puts camera in the scene
        NOTE Assumes body is facing +z direction"""
        cam, camshape = cmds.camera(aspectRatio=self.config['resolution'][0] / self.config['resolution'][1], name=name)
        cmds.setAttr(cam + '.rotate', rotation[0], rotation[1], rotation[2], type='double3')
        cmds.setAttr(camshape + '.renderable', 1)
        # to view the target body
        if obj is not None:
            fitFactor = min(0.8, self.config['resolution'][1] / self.config['resolution'][0])
            cmds.viewFit(cam, obj, f=fitFactor)
            # trans = cmds.getAttr(cam + '.translate')
        return cam, camshape#, trans
    
    def setup_viewfit(self, obj, fitfactor=0.8):
        trans = {}
        for name, val in self.scene["cameras"].items():
            cmds.viewFit(val[0], obj, f=fitfactor)
            translate = cmds.getAttr(val[0] + '.translate')[0]
            trans[name] = translate
        return trans
            

    def query_cam_pos(self, camera_name):
        cam, cam_shape = self.scene["cameras"][camera_name]
        focal_length = cmds.camera(cam_shape, q=True, fl=True)

        inches_to_mm = 25.4
        app_horiz = cmds.camera(cam_shape, q=True, hfa=True) * inches_to_mm
        app_vert = cmds.camera(cam_shape, q=True, vfa=True) * inches_to_mm
        pixel_width = self.config['resolution'][0]
        pixel_height = self.config['resolution'][1]
        focal_length_x_pixel = pixel_width * focal_length / app_horiz
        focal_length_y_pixel = pixel_height * focal_length / app_vert

        translate = cmds.getAttr(cam + ".translate")[0]
        eular_rot = cmds.getAttr(cam + ".rotate")[0]

        # convert_to_opencv_matrix
        K = np.eye(3)
        K[0, 0] = focal_length_x_pixel
        K[1, 1] = focal_length_y_pixel
        K[0, 2] = pixel_width / 2.0
        K[1, 2] = pixel_height / 2.0

        R = utils.eulerAngleToRoatationMatrix((math.radians(eular_rot[0]), math.radians(eular_rot[1]), math.radians(eular_rot[2])))
        return K, R, translate
        # return translate, eular_rot, focal_length_x_pixel, focal_length_y_pixel, pixel_width, pixel_height

    def convert_to_opencv_matrix(self, translate, rotate, fl_x, fl_y, pixel_width, pixel_height):
        K = np.eye(3)
        K[0, 0] = fl_x
        K[1, 1] = fl_y
        K[0, 2] = pixel_width / 2.0
        K[1, 2] = pixel_height / 2.0

        R = utils.eulerAngleToRoatationMatrix((math.radians(rotate[0]), math.radians(rotate[1]), math.radians(rotate[2])))
        return K, R, translate

    def move_floor(self, objs=[]):
        xmins, ymins, zmins, xmaxs, zmaxs = [], [], [], [], [] 
        for obj in objs:
            target_bb = cmds.exactWorldBoundingBox(obj)
            xmins.append(target_bb[0])
            ymins.append(target_bb[1])
            zmins.append(target_bb[2])
            xmaxs.append(target_bb[3])
            zmaxs.append(target_bb[5])
        
        xmin, ymin, zmin, xmax, zmax = min(xmins), min(ymins), min(zmins), max(xmaxs), max(zmaxs)
        cmds.move((xmax + xmin) / 2, ymin - 10, (zmax + zmin) / 2, self.scene["floor"], a=1)
    
    def render_panel(self, cameras, save_to, delta_trans, name=""):
        im_size = self.config['resolution']
        cmds.setAttr("defaultArnoldDriver.aiTranslator", "png", type="string")
        cmds.colorManagementPrefs(e=True, outputTransformEnabled=True, outputUseViewTransform=True)

        # set background as white
        ai_ray_switch = cmds.shadingNode("aiRaySwitch", asUtility=True)
        cmds.setAttr(ai_ray_switch + ".camera", 1, 1, 1)
        cmds.connectAttr("%s.message" % ai_ray_switch, "defaultArnoldRenderOptions.background", f=True)

        for rname, cam_set in cameras.items():
            cam, camshape = cam_set 
            cam_cur_trans = cmds.getAttr(cam + '.translate')[0]
            cmds.setAttr(cam + '.translate', cam_cur_trans[0] + delta_trans[0], cam_cur_trans[1] + delta_trans[1], cam_cur_trans[2] + delta_trans[2])
            local_name = (name + '_' + rname) if name else rname
            filename = os.path.join(os.path.abspath(save_to), local_name )
            cmds.setAttr("defaultArnoldDriver.prefix", filename, type="string")
            arnoldRender(im_size[0], im_size[1], True, True, cam, ' -layer defaultRenderLayer')
            cmds.setAttr(cam + '.translate', cam_cur_trans[0], cam_cur_trans[1], cam_cur_trans[2])


    def render(self, camera_trans, save_to, name='', selected_cameras=None):
        """
            Makes a rendering of a current scene, and saves it to a given path
        """
        im_size = self.config['resolution']
        cmds.setAttr("defaultArnoldDriver.aiTranslator", "png", type="string")
        cmds.colorManagementPrefs(e=True, outputTransformEnabled=True, outputUseViewTransform=True)

        # set background as white
        ai_ray_switch = cmds.shadingNode("aiRaySwitch", asUtility=True)
        cmds.setAttr(ai_ray_switch + ".camera", 1, 1, 1)
        cmds.connectAttr("%s.message" % ai_ray_switch, "defaultArnoldRenderOptions.background", f=True)

        smooth_trans = {}
        for key, trans in camera_trans.items():

            all_trans = np.array(trans)
            smoothed = all_trans.sum(axis=0)/all_trans.shape[0]
            smooth_trans[key] = [float(smoothed[i]) for i in range(smoothed.shape[0])]
        arnoldRender_time=0.0
        cameras = self.scene['cameras']
        for rname, cam_set in cameras.items():
            if selected_cameras is not None and rname not in selected_cameras:
                continue
            cam, camshape = cam_set 
            cam_cur_trans = cmds.getAttr(cam + '.translate')[0]
            cmds.setAttr(cam + '.translate', smooth_trans[rname][0], smooth_trans[rname][1], smooth_trans[rname][2])
            local_name = (name + '_' + rname) if name else rname 
            filename = os.path.join(os.path.abspath(save_to), local_name)
            cmds.setAttr("defaultArnoldDriver.prefix", filename, type="string")
            start=time.time()
            arnoldRender(im_size[0], im_size[1], True, True, cam, ' -layer defaultRenderLayer')
            arnoldRender_time+=time.time()-start
            cam_K, cam_R, cam_T = self.query_cam_pos(rname)
            cam_pos_filename = os.path.join(save_to, local_name + "cam_pos.json")
            # np.savez(cam_pos_filename, cam_K=cam_K, cam_R=cam_R, cam_T=cam_T)
            cam_dict = {"cam_K": cam_K, "cam_R": cam_R, "cam_T":cam_T}
            self._save_json(cam_dict, cam_pos_filename)
            cmds.setAttr(cam + '.translate', cam_cur_trans[0], cam_cur_trans[1], cam_cur_trans[2])
        return arnoldRender_time
    
    def _save_json(self, data, save_path):
        with open(save_path, 'w') as writer:
            json.dump(data, writer, cls=utils.NumpyArrayEncoder)






        




