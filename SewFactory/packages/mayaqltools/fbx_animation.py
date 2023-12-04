# Basic
from __future__ import print_function
from __future__ import division

from random import random, uniform
from re import L
from unittest import skipUnless
import numpy as np
import os
import random
import json

# Maya
from maya import cmds
from maya import OpenMaya
import maya.api.OpenMaya as OM
import pymel.core as pm
pm.loadPlugin("fbxmaya") # LOAD PLUGIN

# Arnold
import mtoa.utils as mutils
from mtoa.cmds.arnoldRender import arnoldRender
import mtoa.core
# import pymel.core as pm
# pm.loadPlugin("fbxmaya") # LOAD PLUGIN

from mayaqltools import utils
reload(utils)

class SmplBody(object):
    """
        Describes the smpl body, with shape and pose blendshapes. 
        support animation and random body shape
        pipeline: 
            1. load body with animation
            2. load skin textures to the body
            3. check if need to extend animation, (less than 72 frames and the last frame is the same with the first frame), if need, extend
            4. add t pose to the animation, t pose at -200 frame and -150 frame (auto key must turn on)
            5. apply pose blend shape
            6. random body shape at the first t pose, re-skeleton the joints
            7. if garment exists, adjust the position according the body at t pose.
        random:
            1. skin textures
            2. body shape, with blend shape values range in [-1, 1]
    """
    def __init__(self, config):

        self.config = config
        self.j_names = {
            0: 'Pelvis',
            1: 'L_Hip', 4: 'L_Knee', 7: 'L_Ankle', 10: 'L_Foot',
            2: 'R_Hip', 5: 'R_Knee', 8: 'R_Ankle', 11: 'R_Foot',
            3: 'Spine1', 6: 'Spine2', 9: 'Spine3', 12: 'Neck', 15: 'Head',
            13: 'L_Collar', 16: 'L_Shoulder', 18: 'L_Elbow', 20: 'L_Wrist',
            14: 'R_Collar', 17: 'R_Shoulder', 19: 'R_Elbow', 21: 'R_Wrist',
        }
        self.joints_mat_path = self.config["joints_mat_path"]
        self.joints_mat = np.load(self.joints_mat_path)

        # specific
        self.body_file = self.config["body_file"]
        self.texture_file = self.config["texture_file"]
        self.animated = self.config["animated"]

        self.before_objs = cmds.ls()
        self.load_body()
        print("total frames: ", self.final_frame)
        self.final_frame = 500
        # scale smpl unit into cm
        self._scale_smpl_body()
        
        self.apply_texture()

        self._center_position()


    def load_body(self, ):
        print("!!! -> bodyfile: {}".format(self.body_file))
        # if finalist
        print("self.animated: {}".format(self.animated))
        if self.animated:
            cmds.file(self.body_file, i=True, type='Fbx', ignoreVersion=True)
            bodyfbx = [bf for bf in cmds.ls(transforms=True) if "bodyfbx2" in bf]
            if len(bodyfbx) == 0:
                bodyfbx = [bf for bf in cmds.ls(transforms=True) if "bodyfbx1" in bf]
            bodyfbx = bodyfbx[0]
            self._collect_body_params(bodyfbx)
        else:
            # if fresh start
            cmds.file(self.body_file, i=True, type='Fbx', ignoreVersion=True, groupReference=True, groupName="bodyfbx")
            bodyfbx = [bf for bf in cmds.ls(transforms=True) if "bodyfbx" in bf][0]
            bodyfbx = cmds.rename(bodyfbx, 'bodyfbx' + '#')
            print("add motion: ".format(self.add_motion(bodyfbx)))
            self.extend_motion(time_stamp=self.config["extend_time_stamp"])

        self.bodyfbx = bodyfbx


    def add_motion(self, bodyfbx):

        self.pose_file = self.config["pose_file"]
        poses, trans, total_frames, gender = utils.load_pose_data(self.pose_file, )
        num_frames = min(self.config["num_frames"], total_frames)
        assert total_frames >= num_frames, "At least {} frames".format(num_frames)
        poses = poses[:num_frames]
        trans = trans[:num_frames]

        self._collect_body_params(bodyfbx)
        
        # anime
        autokey_state = cmds.autoKeyframe(query=True, state=True)
        cmds.autoKeyframe(state=False)
        self._cutkey(self.blendShape, self.skeleton)

        max_bones = poses.shape[1] // 3
        for frame in range(num_frames):
            cmds.currentTime(frame)
            trs = trans[frame] 
            for jidx, j_name in self.j_names.items():
                if jidx + 1 >= max_bones:
                    continue
                bone = '%s_%s' % (self.bone_prefix, j_name)
                c_pose = poses[frame, jidx*3 : jidx*3 +3]
                if 'Pelvis' in bone:
                    cmds.setAttr(bone + '.rotate', float(c_pose[0]),float(c_pose[1]), float(c_pose[2]))
                    cmds.setAttr(bone + '.translate', float(trs[0] * 100), float(trs[2] * 100), float(trs[1] * (-100)))
                    cmds.setKeyframe(bone + '.translate', breakdown=False, controlPoints=False, shape=False)
                else:
                    cmds.setAttr(bone + '.rotate', float(c_pose[0]),float(c_pose[1]), float(c_pose[2]))
                cmds.setKeyframe(bone + '.rotate', breakdown=False, controlPoints=False, shape=False)
                if jidx > 0:
                    cmds.select(bone, replace=True) 
                    real_m = np.array(cmds.xform(query=True, matrix=True)).reshape((4, 4)).T
                    for mi, rot_element in enumerate((real_m[:3, :3] - np.eye(3)).ravel()):
                        bidx = (9 * (jidx - 1)) + mi
                        if bidx < 207:
                            cmds.setAttr('%s.Pose%03d' % (self.blendShape, bidx), rot_element * 1)
                            cmds.setKeyframe('%s.Pose%03d' % (self.blendShape, bidx), breakdown=False, controlPoints=False, shape=False)
        self.final_frame = num_frames
        cmds.autoKeyframe(state=autokey_state)

        if self.config["export_log"]:
            pose_tag = os.path.basename(self.pose_file).split(".")[0] + "_{}".format(num_frames)
            smpl_tag = os.path.basename(self.body_file).split(".")[0]
            save_fn = os.path.join(self.config["export_folder"], pose_tag + "_" + smpl_tag + '_animated.fbx')
            if not os.path.exists(os.path.dirname(save_fn)):
                os.makedirs(os.path.dirname(save_fn))
            self.export_fbx(save_fn)

        return True

    def extend_motion(self, autokey=True, time_stamp=[-150, -200]):
        # add t pose
        try:
            if autokey:
                key_state = cmds.autoKeyframe(query=True, state=True)
                cmds.autoKeyframe(state=True)
                for k_stamp in time_stamp:
                    cmds.currentTime(k_stamp)
                    self._zero_pose()

                # rekey
                if self.config["rekey"]:
                    rekey_stamp = self.config["rekey_stamp"]
                    cmds.currentTime(0)
                    y_rot = cmds.getAttr(self.skeleton[0] + '.rotateY')
                    cmds.currentTime(rekey_stamp)
                    cmds.setAttr(self.skeleton[0] + '.rotate', 0, y_rot, 0)
            if autokey:
                cmds.autoKeyframe(state=key_state)
            
            self.apply_pose_blendshape()
            if self.config["export_final"]:
                pose_tag = os.path.basename(self.pose_file).split(".")[0] + "_{}".format(self.config["num_frames"])
                smpl_tag = os.path.basename(self.body_file).split(".")[0]
                save_fn = os.path.join(self.config["export_folder"], pose_tag + "_" + smpl_tag + '_final.fbx')
                if not os.path.exists(os.path.dirname(save_fn)):
                    os.makedirs(os.path.dirname(save_fn))
                self.export_fbx(save_fn)
        except Exception as e:
            print("extend_motion, exception has occurred, {}".format(e))
    
    def apply_pose_blendshape(self,):
        # get all keyframes
        pelvis_bone = self.skeleton[0]
        ani_attrs2 = cmds.listAnimatable(pelvis_bone)
        if len(ani_attrs2) > 0:
            ani_attr = [ani for ani in ani_attrs2 if 'rotate' in ani][0]
            total_keyframes = cmds.keyframe(ani_attr, query=True, keyframeCount=True)
            times = cmds.keyframe(ani_attr, query=True, index=(0,total_keyframes), timeChange=True)
            for ctime in times:
                cmds.currentTime(ctime)
                for jidx, j_name in self.j_names.items():
                    bone = "{}_{}".format(self.bone_prefix, j_name)
                    if jidx > 0:
                        cmds.select(bone, replace=True)
                        real_m = np.array(cmds.xform(query=True, matrix=True)).reshape((4, 4)).T

                        for mi, rot_element in enumerate((real_m[:3, :3] - np.eye(3)).ravel()):
                            bidx = (9 * (jidx - 1)) + mi
                            if bidx < 207:
                                cmds.setAttr('%s.Pose%03d' % (self.blendShape, bidx), rot_element * 1)
                                cmds.setKeyframe('%s.Pose%03d' % (self.blendShape, bidx), breakdown=False,
                                                                    controlPoints=False, shape=False)
    
    def apply_texture(self, texture_path=""):
        """Ensure Arnold objects are launched in Maya & init GPU rendering settings"""
        objects = cmds.ls('defaultArnoldDriver')

        if not objects:  # Arnold objects not found
            # https://arnoldsupport.com/2015/12/09/mtoa-creating-the-defaultarnold-nodes-in-scripting/
            print('Initialized Arnold')
            mtoa.core.createOptions()
        tex_file = texture_path if os.path.exists(texture_path) else self.texture_file

        default_shader = cmds.shadingNode('aiStandardSurface', asShader=True, name="body_shader")
        default_shader_sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=default_shader + '_sg')
        cmds.connectAttr(default_shader + ".outColor", default_shader_sg + ".surfaceShader")

        if os.path.exists(tex_file): 
            if os.path.isdir(tex_file):
                tex_files = [fn for fn in os.listdir(tex_file) if os.path.splitext(fn)[-1] in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]]
                tex_file = os.path.join(tex_file, random.choice(tex_files))
            else:
                tex_file = tex_file
            base_color_img = cmds.shadingNode('aiImage', asShader=True, name='default_color')
            cmds.setAttr(base_color_img + '.filename', tex_file, type='string')
            cmds.connectAttr(base_color_img + ".outColor", default_shader + ".baseColor")

        else:
            # 0.838, 0.720, 0.658
            color = self.config["texture_color"]
            cmds.setAttr(default_shader + '.baseColor', color[0], color[1], color[2])
        
        cmds.setAttr(default_shader + '.specular', 0)
        cmds.setAttr(default_shader + '.base', 1)
        self.skin_shader, self.skin_shader_sg = default_shader, default_shader_sg
        cmds.sets(self.body_fSMPL, forceElement=self.skin_shader_sg)
    
    
    
    # ===============================================
    def get_mesh_faces(self, body_mesh):
        num_faces = cmds.polyEvaluate(body_mesh, f=True)
        faces = []
        for i in range(num_faces):
            fstr = cmds.polyInfo('{}.f[{}]'.format(body_mesh, i), faceToVertex=True)
            fstr = fstr.split()
            faces.append([int(fstr[2]), int(fstr[3]), int(fstr[4])])
        return faces
    
    def get_mesh_vertices(self, body_mesh):
        vertices_world_position = []
        vertices_index_lst = cmds.getAttr(body_mesh + '.vrts', multiIndices=True)
        for i in vertices_index_lst:
            cur_vert = str(body_mesh) + ".pnts[" + str(i) + ']'
            cur_vert_pos = cmds.xform(cur_vert, query=True, translation=True, worldSpace=True)
            vertices_world_position.append(cur_vert_pos)
        return vertices_world_position

    def save_body_infos(self, save_to='', name=''):
        shape = []
        for i in range(10):
            shape_val = cmds.getAttr(self.blendShape + ".Shape{:03d}".format(i))
            shape.append(shape_val)
        trans = [0, 0, 0]
        poses = []
        for j_idx, j_name in self.j_names.items():
            cur_joint = "{}_{}".format(self.bone_prefix, j_name)
            if 'Pelvis' in cur_joint:
                trans = cmds.getAttr(cur_joint + '.translate')[0]
            c_rot = cmds.getAttr(cur_joint + '.rotate')[0]
            poses.append(c_rot)
        
        save_data = {"shape": np.array(shape), "trans": np.array(trans), "pose": np.array(poses)}

        if save_to != '' and name != '':
            filename = os.path.join(save_to, name + "__body_info.json")
            with open(filename, "w") as writer:
                json.dump(save_data, writer, cls=utils.NumpyArrayEncoder)

        return save_data
    
    def random_skin_textures(self, textures_path):
        tex_file = textures_path if os.path.exists(textures_path) else self.texture_file
        if os.path.exists(tex_file):
            if os.path.isdir(tex_file):
                tex_files = [fn for fn in os.listdir(tex_file) if os.path.splitext(fn)[-1] in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]]
                tex_file = os.path.join(tex_file, random.choice(tex_files))
            else:
                tex_file = tex_file
            color_file = [fn for fn in cmds.listConnections(self.skin_shader) if "default_color" in fn]
            if len(color_file) == 0:
                base_color_img = cmds.shadingNode('aiImage', asShader=True, name='default_color')
                cmds.setAttr(base_color_img + '.filename', tex_file, type='string')
                cmds.connectAttr(base_color_img + ".outColor", self.skin_shader + ".baseColor")
            else:
                color_file = color_file[0]
                cmds.setAttr('{}.filename'.format(color_file), tex_file, type='string')
            return tex_file
        return None

    
    def check_if_t_pose(self):
        num_skes = max(list(self.j_names.keys()))
        for bidx, cj in enumerate(self.skeleton):
            if bidx < num_skes:
                rotation = cmds.getAttr(cj + '.rotate')[0]
                for rot in rotation:
                    if rot != 0:
                        return False
        return True 

    def reskeleton(self):
        # hard coding here, need some extra info to recompute the skeleton
        # Find joints_mat_v*.npz file & return if missing
        # bodymesh, cur_dag = utils.get_mesh_dag(self.body_fSMPL)
        mesh_vertices = self.get_mesh_vertices(self.body_fSMPL)
        gender = 'male' if self.body_fSMPL[0] == 'm' else 'female'

        ## Get new joints
        num_verts_to_use = len(self.joints_mat[gender][1])
        mesh_verts_to_dot = np.asarray(mesh_vertices[:num_verts_to_use])
        subject_j = self.joints_mat[gender].dot(mesh_verts_to_dot)

        ## Lock skinning and set new joint locations
        cmds.skinCluster(self.body_fSMPL, edit=True, moveJointsMode=True)
        for j_idx, j_name in self.j_names.items():
            cur_joint = "{}_{}".format(self.bone_prefix, j_name)
            cmds.move(subject_j[j_idx][0], subject_j[j_idx][1], subject_j[j_idx][2], cur_joint)
        cmds.skinCluster(self.body_fSMPL, edit=True, moveJointsMode=False)

    
    def apply_body_shape(self, body_shape):
        if self.check_if_t_pose():
            key_state = cmds.autoKeyframe(query=True, state=True)
            cmds.autoKeyframe(state=True)
            for rb, bshape in enumerate(body_shape):
                cmds.setAttr(self.blendShape + ".Shape{:03d}".format(rb), bshape)
            self.reskeleton()
            self._center_position()
            cmds.autoKeyframe(state=key_state)
        else:
            print("Failed, can only random body shape at the start t pose")
    
    def random_body_shapes(self, ):
        num_random_bShapes = random.randint(0, 10)
        random_bShapes = random.sample(list(range(10)), num_random_bShapes)
        print("num_random_bShapes={}, random_bShapes={}".format(num_random_bShapes, random_bShapes))
        body_shape = [0] * 10
        for idx in random_bShapes:
            body_shape[idx] = random.uniform(-1.5, 1.5)
        self.apply_body_shape(body_shape)
    
    def palm2fist(self, fist_file):
        currentTime = cmds.currentTime(query=True)
        fist_rotations = json.load(open(fist_file, "r"))
        ori_rotations = self.get_hand_rotations(fist_rotations.keys())

        key_state = cmds.autoKeyframe(query=True, state=True)
        cmds.autoKeyframe(state=True)
        pelvis_bone = self.skeleton[0]
        ani_attrs2 = cmds.listAnimatable(pelvis_bone)
        if len(ani_attrs2) > 0:
            ani_attr = [ani for ani in ani_attrs2 if 'rotate' in ani][0]
            total_keyframes = cmds.keyframe(ani_attr, query=True, keyframeCount=True)
            times = cmds.keyframe(ani_attr, query=True, index=(0,total_keyframes), timeChange=True)
            for ctime in times:
                cmds.currentTime(ctime)
                for key, val in fist_rotations.items():
                    cmds.setAttr(key + ".rotate", val[0], val[1], val[2])
        cmds.autoKeyframe(state=key_state)
        cmds.currentTime(currentTime)
        return ori_rotations
    

    def fist2palm(self, ori_rotations):
        currentTime = cmds.currentTime(query=True)
        key_state = cmds.autoKeyframe(query=True, state=True)
        cmds.autoKeyframe(state=True)
        for ctime, rotations in ori_rotations.items():
            cmds.currentTime(ctime)
            for key, val in rotations.items():
                cmds.setAttr(key + ".rotate", val[0], val[1], val[2])
        cmds.autoKeyframe(state=key_state)
        cmds.currentTime(currentTime)

    
    def get_hand_rotations(self, hand_keys):
        ori_rotations = {}
        pelvis_bone = self.skeleton[0]
        ani_attrs2 = cmds.listAnimatable(pelvis_bone)
        if len(ani_attrs2) > 0:
            ani_attr = [ani for ani in ani_attrs2 if 'rotate' in ani][0]
            total_keyframes = cmds.keyframe(ani_attr, query=True, keyframeCount=True)
            times = cmds.keyframe(ani_attr, query=True, index=(0,total_keyframes), timeChange=True)
            for ctime in times:
                cmds.currentTime(ctime)
                cur_ori_rotations = {}
                for key in hand_keys:
                    cur_ori_rotations[key] = cmds.getAttr(key + ".rotate")[0]
                ori_rotations[ctime] = cur_ori_rotations

            return ori_rotations
        return None

    def export_fbx(self, fbxfile):
        # pm.mel.eval('FBXExport -f "C:/filepath.fbx"')
        # selected = cmds.ls() - self.before_objs
        # cmds.select(selected)
        # pm.mel.FBXExport(f=fbxfile, exportSelected=True)

        selected = [obj for obj in cmds.ls() if obj not in self.before_objs]
        cmds.select(selected)
        # fbxfile = r"C:\Users\seaops\Desktop\tttest.fbx"  
        pm.mel.FBXExport(f=fbxfile, s=True)
    
    def clean(self):
        cmds.delete(self.bodyfbx)
        tl_objs = cmds.ls()
        for i in range(10):
            shape = 'Shape{:03d}'.format(i)
            if shape in tl_objs:
                cmds.delete(shape)
        for i in range(207):
            shape = 'Pose{:03d}'.format(i)
            if shape in tl_objs:
                cmds.delete(shape)
    
    def get_center_pos(self):
        return self.center_pos


    # ===============================================
    
    def _get_associated_deformer(self, shape, deformer_type='blendShape'):
        objset = cmds.listConnections(shape, type='objectSet')
        result = None
        if objset:
            result = cmds.listConnections(objset, type=deformer_type)
            result = result[0]
            return result
        return None

    def _setup_jnames(self, skeleton):
        if len(skeleton) <= 75:
            MODEL_TYPE = 'SMPL'
        elif len(skeleton) <= 156:
            MODEL_TYPE = 'SMPLH'
        elif len(skeleton) <= 165:
            MODEL_TYPE = 'SMPLX'
        else:
            raise TypeError("Model type is not supported")
        print('num_bones: {}, MODEL_TYPE = {}'.format(len(skeleton), MODEL_TYPE))

        if MODEL_TYPE == 'SMPLH':
            self.j_names.update({
                22: 'lindex0', 23: 'lindex1', 24: 'lindex2',
                25: 'lmiddle0', 26: 'lmiddle1', 27: 'lmiddle2',
                28: 'lpinky0', 29: 'lpinky1', 30: 'lpinky2',
                31: 'lring0', 32: 'lring1', 33: 'lring2',
                34: 'lthumb0', 35: 'lthumb1', 36: 'lthumb2',
                37: 'rindex0', 38: 'rindex1', 39: 'rindex2',
                40: 'rmiddle0', 41: 'rmiddle1', 42: 'rmiddle2',
                43: 'rpinky0', 44: 'rpinky1', 45: 'rpinky2',
                46: 'rring0', 47: 'rring1', 48: 'rring2',
                49: 'rthumb0', 50: 'rthumb1', 51: 'rthumb2'
            })
        elif MODEL_TYPE == 'SMPLX':
            self.j_names.update({
                22: 'Jaw', 23: 'L_eye', 24: 'R_eye',
                25: 'lindex0', 26: 'lindex1', 27: 'lindex2',
                28: 'lmiddle0', 29: 'lmiddle1', 30: 'lmiddle2',
                31: 'lpinky0', 32: 'lpinky1', 33: 'lpinky2',
                34: 'lring0', 35: 'lring1', 36: 'lring2',
                37: 'lthumb0', 38: 'lthumb1', 39: 'lthumb2',
                40: 'rindex0', 41: 'rindex1', 42: 'rindex2',
                43: 'rmiddle0', 44: 'rmiddle1', 45: 'rmiddle2',
                46: 'rpinky0', 47: 'rpinky1', 48: 'rpinky2',
                49: 'rring0', 50: 'rring1', 51: 'rring2',
                52: 'rthumb0', 53: 'rthumb1', 54: 'rthumb2'
            })
        else:
            self.j_names.update({
                22: 'L_Hand', 23: 'R_Hand',
            })

        root_bone = [b for b in skeleton if 'root' in b]
        if not root_bone:
            root_bone = [b for b in skeleton if 'Pelvis' in b]
            bone_prefix = root_bone[0].replace('_Pelvis', '')
        else:
            bone_prefix = root_bone[0].replace('_root', '')
        
        return bone_prefix, MODEL_TYPE
    
    def _cutkey(self, blendShape, skeleton):
        for i in range(10):
            cmds.cutKey(blendShape + ".Shape{:03}".format(i))
        for ske in skeleton:
            cmds.cutKey(ske + ".scale")
            if "Pelvis" not in ske:
                cmds.cutKey(ske + '.translate')
    
    def _collect_body_params(self, bodyfbx):
        children = cmds.listRelatives(bodyfbx)
        body_candidate = [ch for ch in children if 'bodyfbx' in ch]
        children = cmds.listRelatives(body_candidate[0] if len(body_candidate) > 0 else bodyfbx)
        model_type = ['SMPLH', "SMPLX", "SMPL"]
        for mt in model_type:
            body_smpl = [ch for ch in children if mt in ch and 'rig' not in ch]
            if len(body_smpl) > 0:
                self.body_fSMPL = body_smpl[0]
                model_type = mt
                break
        # self.body_fSMPL = [ch for ch in children if model_type in ch and 'rig' not in ch][0]
        self.body_rig = [ch for ch in children if model_type in ch and 'rig' in ch][0]
        bodyshape = [ch for ch in cmds.listRelatives(self.body_fSMPL) if model_type + "Shape" in ch][0]
        blendShape = self._get_associated_deformer(bodyshape, 'blendShape')
        skeleton = cmds.listConnections(self._get_associated_deformer(bodyshape, 'skinCluster'), type='joint')
        self.blendShape, self.skeleton = blendShape, skeleton
        self.bone_prefix, self.model_type = self._setup_jnames(self.skeleton)
        if self.animated:
            self.final_frame = self.config["num_frames"] - 1


    
    def _scale_smpl_body(self,):
        # some bug with this opt, not refine yet
        bodymesh = [bd for bd in cmds.listRelatives(self.body_fSMPL) if 'Shape' in bd and 'Orig' not in bd][0]
        bb = cmds.polyEvaluate(bodymesh, boundingBox=True)  # ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        height = bb[1][1] - bb[1][0]
        if height // 100 > 0:
            scale = 1
        elif height // 10 > 0:
            scale = 10
        else:
            scale = 100

        for ske in self.skeleton:
            cmds.setAttr(ske + '.scale', scale, scale, scale)
        bb = cmds.polyEvaluate(bodymesh, boundingBox=True)  # ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        height = bb[1][1] - bb[1][0]
        print('after scale up, the height of the body is ', height)
    
    def _zero_pose(self, ):
        num_skes = max(list(self.j_names.keys()))
        for bidx, cj in enumerate(self.skeleton):
            if bidx < num_skes:
                cmds.setAttr(cj + '.rotate', 0, 0, 0)
                cmds.setKeyframe(cj + '.rotate', breakdown=False, controlPoints=False, shape=False)
                if bidx < 23:
                    for pidx in range(9):
                        cmds.setAttr('%s.Pose%03d' % (self.blendShape, (9 * bidx) + pidx), 0)
                        cmds.setKeyframe('%s.Pose%03d' % (self.blendShape, (9 * bidx) + pidx), breakdown=False, controlPoints=False, shape=False)
   
    def _zero_blenShape_params(self, ):
        for rb in range(10):
            cmds.setAttr(self.blendShape + ".Shape{:03d}".format(rb), 0)
    
    def _center_position(self, ):
        current_time = cmds.currentTime(query=True)
        cmds.currentTime(self.config["extend_time_stamp"][-1])
        bodymesh = [bd for bd in cmds.listRelatives(self.body_fSMPL) if 'Shape' in bd and 'Orig' not in bd][0]
        bb = cmds.polyEvaluate(bodymesh, boundingBox=True)  # ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        self.center_pos = [(bb[0][0] + bb[0][1])/2, (bb[1][0] + bb[1][1])/2, (bb[2][0] + bb[2][1])/2]
        cmds.currentTime(current_time)
        

        