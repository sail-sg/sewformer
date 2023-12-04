from __future__ import print_function
from unittest import skip
from maya import OpenMaya
from maya import cmds, mel
import pymel.core as pm
pm.loadPlugin("fbxmaya") # LOAD PLUGIN

import numpy as np
from datetime import datetime
from glob import glob
import os 
import time
import json
import argparse
from copy import deepcopy
import random

from mayaqltools import utils
reload(utils)

from mayaqltools import fbx_animation

def get_command_args():
    """command line arguments to control the run"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", '-c', help="template config with parameters used for animation", 
                        default=r"meta_infos\configs\anime_config.json")
    parser.add_argument('--base-fbx', '-f', help="initial fbx model used to animate",
                        default=r"meta_infos\fbx_metas\SMPLH_female_010_207.fbx")
    parser.add_argument("--skin-textures", '-t', help="root to save skin textures",
                        default=r"examples\skin_textures")
    parser.add_argument("--pose-root", "-p", help="motion files used to animate fbx",
                        default=r"examples\human_poses")
    parser.add_argument("--output", "-o", help="root to save animated fbxs",
                        default=r"test\posed_fbxs")
    parser.add_argument("--animated", "-a", help="root to animated fbxs (load for check, default to output)",
                        default="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    
    args = get_command_args()
    if not os.path.exists(args.animated):
        # generate posed fbxs
        skin_texture_files = [os.path.join(args.skin_textures, fn) for fn in os.listdir(args.skin_textures) if fn.endswith(".jpg")]
        poses = [os.path.join(args.pose_root, fn) for fn in os.listdir(args.pose_root) if fn.endswith("poses.npz")]
        default_config = json.load(open(args.base_config, "r"))
        for pose in poses:
            c_conf = deepcopy(default_config)
            c_conf["body_file"] = args.base_fbx
            c_conf["texture_file"] = random.sample(skin_texture_files, 1)[0]
            c_conf["pose_file"] = pose 
            c_conf["export_folder"] = args.output
            fbx = fbx_animation.SmplBody(config=c_conf)
            fbx.clean()

    else:
        # show fbx
        # run with running pymel window
        fbxs = [os.path.join(args.animated, fn) for fn in os.listdir(args.animated) if fn.endswith(".fbx")]
        default_config = json.load(open(args.base_config, "r"))
        for fbx_file in fbxs:
            c_conf = deepcopy(default_config)
            c_conf["body_file"] = fbx_file 
            c_conf["animated"] = True
            fbx = fbx_animation.SmplBody(config=c_conf)
            _ = input()
            fbx.clean()
            

