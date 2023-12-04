from __future__ import print_function
import argparse
from copy import deepcopy
from datetime import datetime
from genericpath import exists
import os
import errno
from pickle import NONE
import random
from random import sample
import numpy as np
import sys
import time
import json
from shutil import copyfile
from .garment import Garment

from maya import cmds
import maya.standalone 	

# My modules
import customconfig
# reload in case we are in Maya internal python environment
reload(customconfig)
from mayaqltools import maya_scene, garment_objs, fbx_animation  
from mayaqltools import qualothwrapper as qw
from copy import deepcopy

class GarmentPlayblast(object):
    """
        1. load scene
        2. load smplbody 
        3. load garment
        4. run simulation
        5. run render
    """
    def __init__(self, conf):
        self.config = conf
        self.default_center = [0.037, -29.154, 2.363]
        self.smooth_cameras = {}
        self.max_smooth = 10
    
    def init_scene(self):
        self.config["render"]["config"]["materials"] = self.config["conf"]["mtl_scene_file"]
        self.config["render"]["config"]["textures_path"] = self.config["conf"]["garment_texture_root"]
        self.scene = maya_scene.MayaScene(self.config["render"], clean_on_die=True)
         
    
    def init_sim_props(self):
        render_mtls = self.scene.Mtls.material_types
        sim_props = {}
        if 'default' in render_mtls:
            sim_props['default'] = deepcopy(self.config['sim'])
        for key in render_mtls:
            sim_props[key] = deepcopy(self.config['sim'])
            if key != "default":
                cur_sim_props = json.load(open(os.path.join(self.config['conf']['sim_configs_root'], "mtl_" + key + ".json"), "r"))
                sim_props[key]['config']['material'] = cur_sim_props
        self.sim_props = sim_props
    
    def init_smpl_body(self, fbx_file):
        pose_config = self._make_pose_config(fbx_file)
        self.body = fbx_animation.SmplBody(pose_config)
    
    
    def load_garments(self, props, try_pattern_specs, logger, delete_on_clean):
        garments = {}
        body_center = self.body.get_center_pos()
        delta = [body_center[0] - self.default_center[0], 
                 body_center[1] - self.default_center[1],
                 body_center[2] - self.default_center[2]]
        obstacles = [self.body.body_fSMPL, self.scene.scene['floor']]

        if len(try_pattern_specs) == 1:
            keys = ["one_pieces"]
        else:
            keys = ["bottoms", "tops"]
        
        try:
            for key in keys:
                pattern_file = try_pattern_specs[key]
                mtl_type = sample(list(set(list(self.sim_props.keys())) - set(["default"])), 1)[0]
                c_obstacles = deepcopy(obstacles)
                # for ck, cv in garments.items():
                #     c_obstacles.append(cv.get_qlcloth_geometry())
                garment = garment_objs.MayaGarment(pattern_file, clean_on_die=True)
                garment.load(False, c_obstacles, None, self.sim_props[mtl_type]['config'], delta)
                garments[key] = garment 
                props["garments"]['config']["delta"][key] = delta
                # spec_file = os.path.sep.join(os.path.dirname(os.path.normpath(pattern_file)).split(os.path.sep)[-2:]) 
                props["garments"]['config']["pattern_specs"][key] = pattern_file
                props["garments"]['config']["mtl_specs"][key] = mtl_type
                props["garments"]['config']["load_stats"][key] = True
                obstacles.append(garment.get_qlcloth_geometry())
                if logger is not None: logger.info("Garment Loaded, {}, {}".format(props["garments"]['config']["pattern_specs"][key], mtl_type))
            return props, garments, True
        except Exception as e:
            print(e)
            if logger is not None: logger.info("Garment Load Failed, {}".format(e))
            for key, val in garments.items():
                val.clean(delete_on_clean)
                props["garments"]['config']["load_stats"][key] = False
            return props, garments, False
    

    def simulate_frames(self, props, garments, start, end, logger):
        for frame in range(start, end):
            cmds.currentTime(frame)
            for key, garment in garments.items():
                garment.update_verts_info2()
            self._update_progress(frame + abs(int(self.body.config["extend_time_stamp"][-1])), 
                                  self.body.final_frame + abs(int(self.body.config["extend_time_stamp"][-1])))
            if frame % 20 == 0:
                intersect_self = False
                intersect_others = False
                for key, garment in garments.items():
                    intersect_self = intersect_self or garment.self_intersect_3D(logger=logger)
                    intersect_others = intersect_others or garment.intersect_colliders_3D(logger=logger)
                if intersect_others or intersect_self:
                    props["garments"]['config']["sim_stats"]["stat"] = False
                    props["garments"]['config']["sim_stats"]["frame"] = frame
                    if logger is not None:
                        logger.info("simulate_frames failed by intersection test")
                    return props, False
        return props, True

    
    def simulate(self, props, garments, restore_hands=True, logger=None):
        """
            Setup and run cloth simulator untill static equlibrium is achieved.
            Note:
                * Assumes garment is already properly aligned!
                * All of the garments existing in Maya scene will be simulated
                    because solver is shared!!
                * After the first 20 frames, add a constraint to the body.
        """

        start_time = time.time()
        props = self._init_sim(props, garments, logger)
        if len(garments) == 1:
            keys = ["one_pieces"]
        else:
            keys = ["tops", "bottoms"]

        zero_gravity_steps = props["sim"]["config"]["zero_gravity_steps"] + 9
        attach_delay_steps = props["sim"]["config"]["attach_delay_steps"]
        attach_delay_steps = 1
        if restore_hands:
            rot_file = self.body.config["sim_fist_anime"]
            ori_figure_rotations = self.body.palm2fist(rot_file)
            self.ori_figuer_rotations = ori_figure_rotations

        attach_configs = json.load(open(props["sim"]["config"]["attconf_file"], "r"))

        if logger is not None:
            logger.info("Init animation figures for simulation")
        # import pdb; pdb.set_trace()
        
        start = int(self.body.config["extend_time_stamp"][-1])
        for frame in range(start, start + zero_gravity_steps + attach_delay_steps):
            if len(keys) > 1:
                if frame == start:
                    self._set_gravity(garments[keys[-1]].MayaObjects['solver'], 0)
                elif frame == start + zero_gravity_steps - 1:
                    self._set_gravity(garments[keys[-1]].MayaObjects['solver'], -980)
                elif frame == start + zero_gravity_steps + attach_delay_steps - 1:
                    # att_inputs, att_stiffness, = garments[keys[-1]].check_attach_config()
                    garments[keys[-1]].add_attach_constrains(keys[-1], 
                                                             attach_configs, 
                                                             rate=1, 
                                                             obstacles=self.body.body_fSMPL)
            cmds.currentTime(frame)
            for k,v in garments.items():
                v.update_verts_info2()

            self._update_progress(frame + abs(start), self.body.final_frame + abs(start))  # progress bar

        # run to get tpose stat
        static_frame = int(self.body.config["extend_time_stamp"][0])
        props, stat = self.simulate_frames(props, garments, start + zero_gravity_steps + attach_delay_steps, static_frame, logger)
        if not stat:
            props["garments"]['config']["sim_stats"]["time"] = time.time() - start_time
        else:
            # add constraints to top neck points
            garments[keys[0]].add_attach_constrains(keys[0], attach_configs, rate=0.3, obstacles=self.body.body_fSMPL)
            props, stat = self.simulate_frames(props, garments, static_frame, int(self.body.final_frame), logger)
            if stat:
                if logger is not None:
                    logger.info("SIM Done: run_sim success, time cost: {}, total frames: {}".format(
                        time.time() - start_time, 
                        int(self.body.final_frame) - self.body.config["extend_time_stamp"][-1]))
                props["garments"]['config']["sim_stats"]["time"] = time.time() - start_time
                props["garments"]['config']["sim_stats"]["stat"] = True
        
        return props, stat 

    
    def sim_once(self, props, pattern_spcecs, logger, delete_on_clean):

        skin_texture_fn = self.body.random_skin_textures(
            textures_path=props['conf']['skin_texture_root'] )
        props["body"]["texture"] = skin_texture_fn
        
        self._init_garment_config(props)
        props, garments, load_stat = self.load_garments(
            props, pattern_spcecs, logger, delete_on_clean
        )
        if not load_stat:
            return False
        # save root
        cdir = []
        for key, val in props["garments"]['config']["pattern_specs"].items():
            cdir.append(os.path.basename(os.path.dirname(val)))
        cdir = "__".join(cdir)
        save_path = os.path.join(props["conf"]["output"], cdir)
        props["garments"]["config"]["save_path"] = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        props, stat = self.simulate(props, garments, logger)
        if stat:
            self.save_renders(garments, props, logger)
            props.serialize(os.path.join(props["garments"]["config"]["save_path"], "data_props.json"))

        return props, stat
    
    def save_static_render(self, props, garments, retrieve_props, logger):

        start_time = time.time()
        self.smooth_cameras = {}
        camera_trans = self.scene.setup_viewfit(obj=self.body.body_fSMPL)
        self._update_cameras(camera_trans)
        cmds.refresh(force=True)
        save_to_path = os.path.join(props["garments"]["config"]["save_path"], "static")
        if not os.path.exists(save_to_path):
            os.makedirs(save_to_path)  
        else:
            print("{} exists ...".format(save_to_path))
            return props 

        spec_config = {}
        for key, val in props["garments"]['config']["pattern_specs"].items():
            spec_config[key] = {}
            spec_config[key]["spec"] = val
            spec_config[key]["delta"] = props["garments"]['config']["delta"][key]
            src_spec_fn = spec_config[key]["spec"]
            dst_spec_fn = os.path.join(save_to_path, os.path.basename(os.path.dirname(spec_config[key]["spec"])) + "_specification.json")
            copyfile(src_spec_fn, dst_spec_fn)
        with open(os.path.join(save_to_path, "spec_config.json"), "w") as f:
            json.dump(spec_config, f)

        extr = False
        for key, garment in garments.items():
            mtl_type = props["garments"]['config']["mtl_specs"][key]
            garment.save_mesh(folder=save_to_path, tag='{}'.format(garment.name), static=True) 
            mtl, mtl_sg, random_params = self.scene.Mtls.random_load_materials(mtl_type, pattern_path=props['conf']['garment_texture_root'], extr=extr)
            props["garments"]['config']["render_stats"][key] = {}
            props["garments"]['config']["render_stats"][key]["random_params"] = random_params
            cmds.sets(garment.get_qlcloth_geometry(), forceElement=mtl_sg)
            extr = True
        
        self.scene.render(self.smooth_cameras, save_to_path, 'static')
        self.body.save_body_infos(save_to_path, 'static')
        if logger is not None: logger.info("save_static_pose: {}".format(time.time() - start_time))
        return props

    def save_posed_render(self, props, garments, frame, logger):
        render_path = os.path.join(props["garments"]["config"]["save_path"], "renders")
        if not os.path.exists(render_path):
            os.makedirs(render_path)
        
        body_path = os.path.join(props["garments"]["config"]["save_path"], "poses")
        if not os.path.exists(body_path):
            os.makedirs(body_path)
        
        camera_trans = self.scene.setup_viewfit(obj=self.body.body_fSMPL)
        self._update_cameras(camera_trans)
        cmds.refresh(force=True)
        
        # extr = False
        # for key, garment in garments.items():
        #     mtl_type = props["garments"]['config']["mtl_specs"][key]
        #     mtl, mtl_sg, random_params = self.scene.Mtls.random_load_materials(mtl_type, pattern_path= props['conf']['garment_texture_root'], extr=extr)
        #     props["garments"]['config']["render_stats"][key] = {}
        #     props["garments"]['config']["render_stats"][key]["random_params"] = random_params
        #     cmds.sets(garment.get_qlcloth_geometry(), forceElement=mtl_sg)
        #     extr = True

        frame_string= ('' if frame >= 0 else 'N') + '{:04d}'.format(abs(frame))
        self.scene.render(self.smooth_cameras, render_path, frame_string)
        self.body.save_body_infos(body_path, frame_string)

        points_path = os.path.join(props["garments"]["config"]["save_path"], "objects")
        if not os.path.exists(points_path):
            os.makedirs(points_path)
        
        for key, garment in garments.items():
            garment.save_mesh(folder=points_path, tag='{}_{}'.format(garment.name, frame))  
        if logger is not None: logger.info("save frame : #{}".format(frame))
        return props
        

    
    def save_renders(self, garments, props, restore_hands=True, retrieve_props=False, logger=None):
        
        # restore render figure animations
        self._restore_timeline()
        if restore_hands:
            self.body.fist2palm(self.ori_figuer_rotations)
        self._restore_timeline()

        cmds.hide(self.scene.scene['floor'])

        # save static
        static = int(self.body.config["extend_time_stamp"][0])
        cmds.currentTime(static)
        self.save_static_render(props, garments, retrieve_props, logger)
        if logger is not None: logger.info("SIM: get top and bottom T pose results")

        # save_sequences
        frame_lst = [-50] + list(range(-8, self.body.final_frame + 1, 4))
        for frame in frame_lst:
            cmds.currentTime(frame)
            for key, garment in garments.items():
                garment.update_verts_info2()
            camera_trans = self.scene.setup_viewfit(obj=self.body.body_fSMPL)
            self._update_cameras(camera_trans)
            # if frame % props["render"]["config"]["shape_skip"] == 0:
            props = self.save_posed_render( props, garments, frame, logger)
        cmds.showHidden(self.scene.scene['floor'])
    
    def load_all_garments(self, data_paths):
        glists = {}
        file_root = data_paths["file_root"]
        sub_patterns = data_paths["sub_patterns"]
        for key, val in sub_patterns.items():
            glists[key] = []
            for fpath in val:
                for fdir in os.listdir(os.path.join(file_root, fpath)):
                    if os.path.isdir(os.path.join(file_root,fpath, fdir)):
                        glists[key].append(Garment(fdir, os.path.join(file_root, fpath)))
        return glists

    def pair_fbx_garments(self, fbxs, garment_lists):
        keys = ["tops", "bottoms"]
        num_pairs = min(len(garment_lists[keys[0]]), len(garment_lists[keys[1]]))
        pairs = list(zip(random.sample(garment_lists[keys[0]], num_pairs), random.sample(garment_lists[keys[1]], num_pairs)))
        garment_lists = pairs + garment_lists["one_pieces"]
        num_final_pairs = min(len(fbxs), len(garment_lists))
        pairs = list(zip(random.sample(fbxs, num_final_pairs), random.sample(garment_lists, num_final_pairs)))
        return pairs

    def adjust_scene_objs(self):
        obj = [ch for ch in cmds.listRelatives(self.body.body_fSMPL) if "Shape" in ch][0]
        self.scene.move_floor(objs=[obj])
        trans = self.scene.setup_viewfit(obj=obj)
        return trans


    def generate(self, fbxs, garment_paths, logger=None):
        qw.load_plugin()
        self.init_scene()
        if logger is not None: logger.info("init scene done")
        self.init_sim_props()
        if logger is not None: logger.info("init sim props done")
        
        garment_lists = self.load_all_garments(garment_paths)
        fbx_garment_pairs = self.pair_fbx_garments(fbxs, garment_lists)
        for pair in fbx_garment_pairs:
            fbx_file, garments_file = pair[0], pair[1]
            c_props = deepcopy(self.config)
            c_props["body"]["name"] = fbx_file
            c_props["render"]["config"]["shape_skip"] = 4
            self.ori_figuer_rotations = None
            self.init_smpl_body(fbx_file)
            if logger is not None:
                logger.info("Loaded Ani Name: {}".format(fbx_file))
            self._restore_timeline()
            self.body.random_body_shapes()
            self.smooth_cameras = {}

            body_info = self.body.save_body_infos()
            if logger is not None:
                logger.info("Body Shape: {}".format(body_info['shape'].tolist()))
            c_props["body"]["shape"] = body_info['shape'].tolist()
            # todo check if need to be changed
            c_props["body"]["total_frames"] = self.body.final_frame
            camera_trans = self.adjust_scene_objs()
            self._update_cameras(camera_trans)

            c_pattern_specs = {}
            if isinstance(garments_file, tuple):
                c_pattern_specs = {
                    "tops": garments_file[0].to_spec_path(data_root=""),
                    "bottoms": garments_file[1].to_spec_path(data_root="")
                }
            else:
                c_pattern_specs["one_pieces"] = garments_file.to_spec_path(data_root="")
                
            stat = self.sim_once(c_props, c_pattern_specs, logger, delete_on_clean=True)
            if not stat:
                if logger is not None: logger.info("# Simulation FAIL")
            self.body.clean()
            
        return

        

    # ------- Self-Utils ---------
    def _init_garment_config(self, props):
           if "garment" not in props:
                props.set_section_config(
                "garments",
                pattern_specs={},
                mtl_specs={},
                delta={},
                load_stats={},
                sim_stats={},
                render_stats={},
                save_path=""
            )
    
    
    def _init_sim(self, props, garments, logger):
        """
            Basic simulation settings before starting simulation
        """
        for key, garment in garments.items():
            cmds.setAttr(garment.get_qlcloth_props_obj() + '.active', 1)
            cmds.setAttr(garment.MayaObjects["solver"] + '.active', 1)
        if len(garments) == 1:
            keys = ["one_pieces"]
        else:
            keys = ["bottoms", "tops"]
        for idx, key in enumerate(keys):
            if "sim_stats" not in props["garments"]['config']: props["garments"]['config']["sim_stats"] = {}
            props["garments"]['config']["sim_stats"][key] = {}
            cmds.setAttr(garments[key].MayaObjects['solver'] + ".selfCollision", 1)
            if idx == 0:
                cmds.setAttr(garments[key].MayaObjects['solver'] + ".frameSamples", 1)
                props["garments"]['config']["sim_stats"][key]["framsamples"] = 2
            else:
                cmds.setAttr(garments[key].MayaObjects['solver'] + ".frameSamples", 2)
                props["garments"]['config']["sim_stats"][key]["framsamples"] = 2

            if idx == 0:
                cmds.setAttr(garments[key].MayaObjects["solver"] + '.startTime', self.body.config["extend_time_stamp"][-1])
            else:
                cmds.setAttr(garments[key].MayaObjects["solver"] + '.startTime', self.body.config["extend_time_stamp"][-1] + 10)

            cmds.setAttr(garments[key].MayaObjects["solver"] + '.solverStatistics', 0)

        cmds.playbackOptions(ps=0, min=self.body.config["extend_time_stamp"][-1], max=self.body.final_frame)

        if logger is not None: logger.info("Init playblast ..")
        return props
    
    def _set_gravity(self, solver, gravity):
        """Set a given value of gravity to sim solver"""
        cmds.setAttr(solver + '.gravity1', gravity)
    
    def _update_progress(self, progress, total):
        """Progress bar in console"""
        # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        amtDone = progress / (total * 1.0)
        num_dash = int(amtDone * 50)
        sys.stdout.write('\rProgress: [{0:50s}] {1:.1f}%\r'.format('#' * num_dash + '-' * (50 - num_dash), amtDone * 100))
        sys.stdout.flush()
    
    def _update_cameras(self, camera_trans):
        for key, trans in camera_trans.items():
            if key not in self.smooth_cameras:
                self.smooth_cameras[key] = [trans]
            else:
                if len(self.smooth_cameras[key]) >= self.max_smooth:
                    self.smooth_cameras[key] = self.smooth_cameras[key][1:]
                self.smooth_cameras[key].append(trans)
    
    def _restore_timeline(self):
        cmds.playbackOptions(ps=0, min=int(self.body.config["extend_time_stamp"][-1]), max=self.body.final_frame)  # 0 playback speed = play every frame
        cmds.currentTime(int(self.body.config["extend_time_stamp"][-1]))
    
    def _make_pose_config(self, fbx_file):
        anime_base_config = self.config['conf']["fbxs_base_config"]
        default_config = json.load(open(anime_base_config, "r"))
        c_conf = deepcopy(default_config)
        c_conf["body_file"] = fbx_file
        c_conf["animated"] = True
        return c_conf
    


class PredictPlayblast(GarmentPlayblast):
    def __init__(self, conf):
        super(PredictPlayblast, self).__init__(conf)
        self.set_panel_render_camera()
    
    def set_panel_render_camera(self):
        self.panel_cameras = {}
        cam, camshape = cmds.camera(aspectRatio=1, name="panel_front")
        cmds.setAttr(cam + '.rotate', -18.338, 25.8, 0, type='double3')
        cmds.setAttr(cam + '.translate', 145.706, 195.8, 377.989)
        cmds.setAttr(camshape + '.renderable', 1)
        self.panel_cameras["panel_front"] = (cam, camshape)
        cam, camshape = cmds.camera(aspectRatio=1, name="panel_back")
        cmds.setAttr(cam + '.rotate', -18.338, 205.8, 0, type='double3')
        cmds.setAttr(cam + '.translate', -165.706, 185.079, -263.00)
        cmds.setAttr(camshape + '.renderable', 1)
        self.panel_cameras["panel_back"] = (cam, camshape)
        self.default_pelvis_center = [-28.89, 68.303, 57.545]
    
    def run_pred_panel_render(self, garments, pred_folder, logger=None):
        cmds.refresh(force=True)
        save_to_path = os.path.join(pred_folder, "panels")
        if not os.path.exists(save_to_path):
            os.makedirs(save_to_path) 
        
        extr = False
        for key, garment in garments.items():
            mtl, mtl_sg, _ = self.scene.Mtls.random_load_materials("default", color=key, extr=extr)
            # cmds.sets(garment.get_qlcloth_geometry(), forceElement=mtl_sg)
            if 'qlClothOut' not in garment.MayaObjects:
                children = cmds.listRelatives(garment.MayaObjects['pattern'], ad=True)
                cloths = [obj for obj in children if 'qlCloth' in obj and 'Out' in obj and 'Shape' not in obj]
                for cloth in cloths:
                    print(cloth)
                    cmds.sets(cloth, forceElement=mtl_sg)
            extr = True
        cmds.hide(self.body.body_fSMPL)
        cmds.hide(self.scene.scene['floor'])
        cur_pelvis_center = cmds.joint(self.body.skeleton[0], p=True, q=True)
        delta = [cur_pelvis_center[i] - self.default_pelvis_center[i] for i in range(3)]
        self.scene.render_panel(self.panel_cameras, save_to_path, delta, "panels")
        cmds.showHidden(self.body.body_fSMPL)
        cmds.showHidden(self.scene.scene['floor'])
        if logger is not None: logger.info("Render panels done .")


    def split_predicted_specs(self, specification, logger=False):
        def check_splits(specification):
            stitches = specification["pattern"]["stitches"]
            total_panels = []
            for stitch in stitches:
                total_panels.append(stitch[0]["panel"])
                total_panels.append(stitch[1]["panel"])
            total_panels = sorted(list(set(total_panels)))
            parents = {}
            for stitch in stitches:
                side1, side2 = stitch
                if side1["panel"] not in parents:
                    parents[side1["panel"]] = side1["panel"]
                if side2["panel"] not in parents:
                    parents[side2["panel"]] = side1["panel"]

            def find(key):
                while parents[key] != key:
                    parents[key] = find(parents[key])
                    key = parents[key]
                return key
            
            for stitch in stitches:
                key1, key2 = stitch[0]["panel"], stitch[1]["panel"]
                p1 = find(key1)
                p2 = find(key2)
                if p1 != p2:
                    parents[p2] = p1 
            
            sub_groups = {}
            for key, val in parents.items():
                if parents[val] not in sub_groups:
                    sub_groups[parents[val]] = [key]
                else:
                    sub_groups[parents[val]].append(key)
            return sub_groups
        
        sub_groups = check_splits(specification)
        num_stitch_panels = sum([len(val) for key, val in sub_groups.items()])
            
        if num_stitch_panels != len(specification["pattern"]["panels"]):
            if logger is not None:
                logger.info("stitches and panels are not consistent")
            else:
                print("stitches and panels are not consistent")
            return None

        if len(sub_groups) == 1:
            return {"one_pieces": specification}
        else:
            sub_specs = {}
            heights = {}
            for key, val in sub_groups.items():
                sub_spec = {}
                height = []
                sub_spec["properties"] = specification["properties"]
                sub_spec["parameters"] = specification["parameters"]
                sub_spec["parameter_order"] = specification["parameter_order"]
                sub_spec["pattern"] = dict.fromkeys(specification["pattern"])
                cnt = 0
                for panel in specification["pattern"]["panels"]:
                    if panel in val:# or check_in(panel, panel_cls[key]):
                        if sub_spec["pattern"]["panels"] is None: sub_spec["pattern"]["panels"] ={}
                        sub_spec["pattern"]["panels"][panel] = specification["pattern"]["panels"][panel]
                        if sub_spec["pattern"]["panel_order"] is None: sub_spec["pattern"]["panel_order"] = []
                        sub_spec["pattern"]["panel_order"].append(panel)
                        if sub_spec["pattern"]["new_panel_ids"] is None: sub_spec["pattern"]["new_panel_ids"] = []
                        sub_spec["pattern"]["new_panel_ids"].append(cnt)
                        cnt += 1
                        height.append(specification["pattern"]["panels"][panel]["translation"][1])
                for stitch in specification["pattern"]["stitches"]:
                    side1, side2 = stitch
                    if side1["panel"] in val and side2["panel"] in val:
                        if sub_spec["pattern"]["stitches"] is None: sub_spec["pattern"]["stitches"]= []
                        sub_spec["pattern"]["stitches"].append(stitch)
                sub_specs[key] = sub_spec
                heights[key] = sum(height) / len(height)
            
            ks = list(heights.keys())
            if heights[ks[0]] > heights[ks[1]]:
                return {"tops": sub_specs[ks[0]], "bottoms": sub_specs[ks[1]]}
            else:
                return {"tops": sub_specs[ks[1]], "bottoms": sub_specs[ks[0]]}
    
    def _get_pattern_spec(self, pred_folder, logger):
        if os.path.isfile(pred_folder):
            pred_specs = [pred_folder]
        else:
            pred_specs = [os.path.join(pred_folder, fn) for fn in os.listdir(pred_folder)
                            if fn.endswith("specification.json") and "predicted" in fn]
        if len(pred_specs) == 0:
            if logger is not None: logger.info("No predictions saved in {}".format(pred_folder))
            else: print("No predictions saved in {}".format(pred_folder))
        try:
            pred_spec = self.split_predicted_specs(json.load(open(pred_specs[0], "r")))
            if pred_spec is None:
                if logger is not None: logger.info("stitches and panels are not consistent, {}".format(pred_folder))
                else: print("stitches and panels are not consistent, {}".format(pred_folder))
            for key, spec in pred_spec.items():
                with open(os.path.join(pred_folder, "predicted_specification_{}.json".format(key)), "w") as f:
                    json.dump(spec, f, indent=2)
                    pred_spec[key] = os.path.join(pred_folder, "predicted_specification_{}.json".format(key))
            return pred_spec
        except Exception as e:
            if logger is not None: logger.info("Fail in splitting specification, {}".format(pred_folder))
            else: print("Fail in splitting specification, {}".format(pred_folder))
            

    
    def _make_pose_config(self, pred_folder, restore=False):
        if not restore:
            fbx_file = [fn for fn in os.listdir(pred_folder) if fn.endswith("_final.fbx")]
        
            if len(fbx_file) > 0:
                fbx_file = fbx_file[0]
                anime_base_config = self.config['conf']["fbxs_base_config"]
                default_config = json.load(open(anime_base_config, "r"))
                c_conf = deepcopy(default_config)
                pose_file = os.path.join(self.config["conf"]["pose_root"], os.path.basename(pred_folder) + ".json")
                c_conf['pose_file'] = pose_file
                c_conf["body_file"] = os.path.join(pred_folder, fbx_file)
                c_conf["animated"] = True
                c_conf["export_final"] = False
                c_conf["texture_file"] = ""
                fbx_data = json.load(open(c_conf['pose_file']))
                smpl_shape = fbx_data["shape_tuned"] if "shape_tuned" in fbx_data else fbx_data["shape"]
                return c_conf, smpl_shape
            else:
                anime_base_config = self.config['conf']["fbxs_base_config"]
                default_config = json.load(open(anime_base_config, "r"))
                c_conf = deepcopy(default_config)
                conf = self.config["conf"]
                pred_name = os.path.basename(pred_folder)
                pose_file = os.path.join(conf["pose_root"], pred_name + ".json")
                c_conf['pose_file'] = pose_file
                c_conf['num_frames'] = 500
                c_conf["export_folder"] = pred_folder
                c_conf["body_file"] = conf["body_file"]
                c_conf["joints_mat_path"] = conf["joint_mat"]
                c_conf["animated"] = False
                c_conf["texture_file"] = ""     # use default color
                return c_conf, [0] * 10
        else:
            spec_file = os.path.join(pred_folder, "spec_config.json")
            data_spec = json.load(open(spec_file, "r"))
            fbx_file = os.path.join(self.config['conf']['fbxs_root'], data_spec["pose_file"])
            anime_base_config = self.config['conf']["fbxs_base_config"]
            default_config = json.load(open(anime_base_config, "r"))
            c_conf = deepcopy(default_config)
            c_conf["body_file"] = os.path.join(pred_folder, fbx_file)
            c_conf["animated"] = True
            c_conf["export_final"] = False
            c_conf["texture_file"] = ""
            smpl_shape = data_spec["shape"]
            return c_conf, smpl_shape


    def retieve_props(self, pred_folder, restore=False):
        c_conf, smpl_shape = self._make_pose_config(pred_folder, restore=restore)
        self.body = fbx_animation.SmplBody(config=c_conf)
        # import pdb; pdb.set_trace()
        self._restore_timeline()
        self.body.apply_body_shape(smpl_shape)
        return c_conf, smpl_shape


    def simulate_predictions(self, predictions, retrieve_props=False, logger=None):
        qw.load_plugin()
        self.init_scene()
        self.init_sim_props()
        import pdb; pdb.set_trace()

        for pred_folder in predictions:
            
            # if os.path.exists(os.path.join(pred_folder, "panels")): continue
            try: 
                pred_spec = self._get_pattern_spec(pred_folder, logger)
                # pass
            except Exception as e:
                print(e)
                continue
            else:
                c_props = deepcopy(self.config)
                # load smpl
                body_conf, smpl_shape = self.retieve_props(pred_folder, retrieve_props)
                body_info = self.body.save_body_infos()
                if logger is not None:
                    logger.info("Body Shape: {}".format(body_info['shape'].tolist()))
                c_props["body"]["shape"] = body_info['shape'].tolist()
                c_props["body"]["total_frames"] = self.body.final_frame

                trans = self.adjust_scene_objs()
                self._update_cameras(trans)

                # load garments
                self._init_garment_config(c_props)
                
            
                # if "dress" in pred_folder or "jumpsuit" in pred_folder: 
                #     keys = ["gt_one_pieces"]
                #     pred_spec = {keys[0]: pred_folder}
                # elif "tee" in pred_folder or "jacket" in pred_folder: 
                #     keys = ["gt_tops"]
                #     pred_spec = {keys[0]: pred_folder}
                # else: 
                #     keys = ["gt_bottoms"]
                #     pred_spec = {keys[0]: pred_folder}

                if len(pred_spec) == 1:
                    keys = ["one_pieces"]
                else:
                    keys = ["bottoms", "tops"] 

                
                garments = {}
                body_center = self.body.get_center_pos()
                delta = [body_center[0] - self.default_center[0], 
                        body_center[1] - self.default_center[1],
                        body_center[2] - self.default_center[2]]
                obstacles = [self.body.body_fSMPL, self.scene.scene['floor']]

                # load garments
                for key in keys:
                    garment = garment_objs.MayaGarment(pred_spec[key], clean_on_die=True)
                    garment.load(rest_delay=True, delta_trans=delta)
                    garments[key] = garment
                    c_props["garments"]['config']["delta"][key] = delta
                    c_props["garments"]['config']["pattern_specs"][key] = pred_spec[key]
                
                # run panel render
                save_path = pred_folder.split(".")[0]
                c_props["garments"]["config"]["save_path"] = save_path

                # self.run_pred_panel_render(garments, save_path, logger=logger)
                self.run_pred_panel_render(garments, pred_folder, logger=logger)
                # for key in keys:
                #     garments[key].clean()

                # load rest and simulate 
                for key in keys:
                    c_obstacles = deepcopy(obstacles)
                    garments[key].load_rest(obstacles=c_obstacles, shader_group=None, config=self.sim_props["cotton"]["config"])
                    c_props["garments"]['config']["mtl_specs"][key] = "default"
                    c_props["garments"]['config']["load_stats"][key] = True
                    obstacles.append(garments[key].get_qlcloth_geometry())
                    if logger is not None: logger.info("Garment Loaded, {}, {}".format(c_props["garments"]['config']["pattern_specs"][key], "default"))

                c_props, stat = self.simulate(c_props, garments, restore_hands=False, logger=logger)
                if stat:
                    self.save_renders(garments, c_props, restore_hands=False, retrieve_props=retrieve_props, logger=logger)
                    c_props.serialize(os.path.join(c_props["garments"]["config"]["save_path"], "data_props.json"))

                if not stat:
                    if logger is not None: logger.info("# Simulation FAIL")
                self.body.clean()
    
    
    def restore_render_file(self, garments, retrieve_props, props):
        # for deep fashion
        if retrieve_props:
            restore_file = os.path.join(props["garments"]["config"]["save_path"], "spec_config.json")
            restore_file = json.load(open(restore_file, "r"))
            render_files = restore_file["render"]
            return render_files
            
        else:
            restore_file = os.path.join(props["garments"]["config"]["save_path"], "color.json")
            colors = json.load(open(restore_file, "r")) if os.path.exists(restore_file) else {"one_pieces": [128, 128, 128]}
            if len(garments) == len(colors):
                return colors
            elif len(garments) < len(colors):
                colors = {"one_pieces": colors["tops"]}
            else:
                colors = {"tops": colors["one_pieces"],
                        "bottoms": colors["one_pieces"]}
            return colors

    
    def save_static_render(self, props, garments, retrieve_props, logger):
        start_time = time.time()
        self.smooth_cameras = {}
        camera_trans = self.scene.setup_viewfit(obj=self.body.body_fSMPL)
        self._update_cameras(camera_trans)
        cmds.refresh(force=True)
        save_to_path = os.path.join(props["garments"]["config"]["save_path"], "static")
        if not os.path.exists(save_to_path):
            os.makedirs(save_to_path)  
        
        spec_config = {}
        for key, val in props["garments"]['config']["pattern_specs"].items():
            spec_config[key] = {}
            spec_config[key]["spec"] = val
            spec_config[key]["delta"] = props["garments"]['config']["delta"][key]
        with open(os.path.join(save_to_path, "spec_config.json"), "w") as f:
            json.dump(spec_config, f)
        
        # TODO
        # import pdb; pdb.set_trace()
        extr = False
        restore_config = self.restore_render_file(garments, retrieve_props, props)
        for key, garment in garments.items():
            if retrieve_props:
                mtl_type = restore_config[key]["mtl_specs"]
                pattern_path = r"D:\garmentdataset\fabric_texture"
                mtl, mtl_sg, _ = self.scene.Mtls.restore_materials(mtl_type, pattern_path=pattern_path, restore_config=restore_config[key]["random_params"], extr=extr)
            else:
                mtl_type = props["garments"]['config']["mtl_specs"][key]
                mtl, mtl_sg, _ = self.scene.Mtls.restore_materials(mtl_type, restore_config=restore_config[key], extr=extr)
            props["garments"]['config']["render_stats"][key] = {}
            props["garments"]['config']["render_stats"][key]["random_params"] = restore_config
            cmds.sets(garment.get_qlcloth_geometry(), forceElement=mtl_sg)

            garment.save_mesh(folder=save_to_path, tag='{}'.format(garment.name), static=True) 
            extr = True
        
        self.scene.render(self.smooth_cameras, save_to_path, 'static')
        self.body.save_body_infos(save_to_path, 'static')
        if logger is not None: logger.info("save_static_pose: {}".format(time.time() - start_time))
        return props
        
            
        
        

            


                
            
            




        



