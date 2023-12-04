"""
    Run or Resume simulation of a pattern dataset with MayaPy standalone mode
    Note that this module is executed in Maya (or by mayapy) and is Python 2.7 friendly.

    How to use: 
        * fill out system.json with approppriate paths 
        Running itself:
        <path_to_maya/bin>/mayapy.exe ./datasim.py --data <dataset folder name> --minibatch <size>  --config <simulation_rendering_configuration.json>

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
                        default="deepfashion")
    parser.add_argument('--config', '-c', help="config file for dataset resource", 
                        default="meta_infos\\configs\\data_sim_configs.json")
    parser.add_argument('--sim-template', '-t', help="template props (predicted)",
                        default="meta_infos\\sim_configs\\template.json")
    parser.add_argument('--predicted-root', '-r', help="path root to predicted results",
                        default="other_scripts\\deepfahsion_results\\hats")
    parser.add_argument('--fbx-root', '-f', help="path root of posed fbxs (fbx or smpl info)", 
                        default="")
    parser.add_argument('--base-fbx', '-b', help="initial fbx model used to animate",
                        default="meta_infos\\fbx_metas\\SMPL_female_010_207.fbx")
    parser.add_argument('--joint-mat', '-j', help="",
                        default="meta_infos\\fbx_metas\\joints_mat_SMPL.npz")
    parser.add_argument('--attach-conf', '-a', help="config file for attachments to the human body",
                        default="meta_infos\\configs\\attach_configs_predicted.json")
    parser.add_argument('--pose-root', '-p', help="root to estimated smpl pose files for deep fashion.",
                        default="D:\\liulj_dev\\deepfashion\\spin_smpl_np")
    parser.add_argument('--output', '-o', help="save path for simulated dataset (NONE means save in resouce file)",
                        default="examples\\synthesis")

    args = parser.parse_args()
    config = json.load(open(args.config, "r"))
    config["fbxs_root"] = args.fbx_root
    config["pose_root"] = args.pose_root
    config["body_file"] = args.base_fbx
    config["joint_mat"] = args.joint_mat
    config["output"] = args.output 
    config["dataname"] = args.dataname
    # config["attach_conf"] = args.attach_conf
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
    if os.path.exists(args.config["fbxs_root"]):
        fbxs = [os.path.join(args.config["fbxs_root"], fn) for fn in os.listdir(args.config["fbxs_root"]) 
                if fn.endswith("_final.fbx")]
        if len(fbxs) == 0:
            print("start from creating fbx from pose files")
    else:
        fbxs = []
    predictions = []
    for fn in os.listdir(args.predicted_root):
        if os.path.isdir(os.path.join(args.predicted_root, fn)):
            predictions.append(os.path.join(args.predicted_root, fn))
    print(predictions)

    # predictions = [os.path.join(args.predicted_root, fn) for fn in os.listdir(args.predicted_root) 
    #                if os.path.exists(os.path.join(args.predicted_root, fn)) and os.path.isdir(os.path.join(args.predicted_root, fn))]
    
    # ------- Props ---------
    props = customconfig.Properties()
    props.set_basic(conf=args.config,
                    name=args.dataname,
                    size=0,
                    to_subfolders=True,
                    body_sum=len(fbxs))

    if os.path.exists(args.sim_template):
        if os.path.exists(args.attach_conf):
            props.merge(args.sim_template)
            props['sim']['config']['attconf_file'] = args.attach_conf

    # init maya 
    init_mayapy()
    import mayaqltools as mymaya  # has to import after maya is loaded
    reload(mymaya)  # reload in case we are in Maya internal python environment

    logging.basicConfig(filename= os.path.join(os.getcwd(), r'playblast_{}_run.txt'.format(args.dataname)), 
                    filemode='w', level=logging.INFO,
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p')

    logger = logging.getLogger(__name__)

    logger.info("Start running {}".format(args.dataname))
    playblast = mymaya.playblast.PredictPlayblast(conf=props)
    playblast.simulate_predictions(predictions, retrieve_props=args.dataname=="sewfactory", logger=logger)





