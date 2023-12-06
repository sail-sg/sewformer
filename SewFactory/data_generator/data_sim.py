"""
    Run or Resume simulation of a pattern dataset with MayaPy standalone mode
    Note that this module is executed in Maya (or by mayapy) and is Python 2.7 friendly.
"""

from __future__ import print_function
import argparse
from copy import deepcopy
import os
import json
import logging

from maya import cmds
import maya.standalone 	

import customconfig
reload(customconfig)


def get_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', '-n', help="name of generated",
                        default="sewfactory")
    parser.add_argument('--config', '-c', help="config file for dataset resource", 
                        default="meta_infos\\configs\\data_sim_configs.json")
    parser.add_argument('--sim-template', '-t', help="template props",
                        default="meta_infos\\sim_configs\\template.json")
    parser.add_argument('--fbx-root', '-f', help="path root of posed fbxs", 
                        default="test\\posed_fbxs")
    parser.add_argument('--output', '-o', help="save path for simulated dataset (NONE means save in resouce file)",
                        default="test\\synthesis")
    # parser.add_argument('')

    args = parser.parse_args()
    config = json.load(open(args.config, "r"))
    config["fbxs_root"] = args.fbx_root
    config["output"] = args.output 
    args.config = config
    return args

def init_mayapy():
    try: 
        print('Initilializing Maya tools...')
        maya.standalone.initialize()
        print('Load plugins')
        cmds.loadPlugin('mtoa.mll')  # https://stackoverflow.com/questions/50422566/how-to-register-arnold-render
        cmds.loadPlugin('objExport.mll')  # same as in https://forums.autodesk.com/t5/maya-programming/invalid-file-type-specified-atomimport/td-p/9121166
        cmds.loadPlugin('fbxmaya.mll')
    except Exception as e: 
        print(e)
        pass

def stop_mayapy():
    maya.standalone.uninitialize() 
    print("Maya stopped")


if __name__ == '__main__':
    args = get_command_args()
    fbxs = [os.path.join(args.config["fbxs_root"], fn) for fn in os.listdir(args.config["fbxs_root"]) 
            if fn.endswith("_final.fbx")]
    
    # ------- Props ---------
    props = customconfig.Properties()
    props.set_basic(conf=args.config,
                    name=args.dataname,
                    size=0,
                    to_subfolders=True,
                    body_sum=len(fbxs))

    if os.path.exists(args.sim_template):
        props.merge(args.sim_template)

    # init maya 
    init_mayapy()
    import mayaqltools as mymaya  # has to import after maya is loaded
    reload(mymaya)  # reload in case we are in Maya internal python environment

    logging.basicConfig(filename= os.path.join(os.getcwd(), r'playblast_run.txt'), 
                    filemode='w', level=logging.INFO,
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p')

    logger = logging.getLogger(__name__)


    logger.info("Start running")
    playblast = mymaya.playblast.GarmentPlayblast(conf=props)
    playblast.generate(fbxs, args.config["patterns"], logger)



    




