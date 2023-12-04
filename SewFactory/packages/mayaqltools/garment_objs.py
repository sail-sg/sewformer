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

# Maya
from maya import cmds
from maya import OpenMaya
import maya.api.OpenMaya as OM

# Arnold
import mtoa.utils as mutils
from mtoa.cmds.arnoldRender import arnoldRender
import mtoa.core

# My modules
import pattern.core as core
from mayaqltools import qualothwrapper as qw
from mayaqltools import utils
reload(core)
reload(qw)
reload(utils)

class PatternLoadingError(BaseException):
    """To be rised when a pattern cannot be loaded correctly to 3D"""
    pass

class MayaGarment(core.ParametrizedPattern):
    """
    Extends a pattern specification in custom JSON format to work with Maya
        Input:
            * Pattern template in custom JSON format
        * import panel to Maya scene TODO
        * cleaning imported stuff TODO
        * Basic operations on panels in Maya TODO
    """
    def __init__(self, pattern_file=None, clean_on_die=False):
        super(MayaGarment, self).__init__(pattern_file)

        self.self_clean = clean_on_die

        self.loaded_to_maya = False
        self.last_verts = None
        self.current_verts = None
        self.obstacles = []
        self.shader_group = None
        self.MayaObjects = {}
        self.config = {
            'material': {},
            'body_friction': 0.5, 
            'resolution_scale': 8
        }
    
    def __del__(self):
        """Remove Maya objects when dying"""
        if self.self_clean:
            print("if self.self_clean:")
            self.clean(True)
    
    def load(self, rest_delay=False, obstacles=[], shader_group=None, config={},
             delta_trans=[0, 0, 0], parent_group=None):
        """
            Loads current pattern to Maya as simulatable garment.
            If already loaded, cleans previous geometry & reloads
            config should contain info on fabric matereials & body_friction (collider friction) if provided

            rest_delay: True if only load panels, False if load full garment

        """
        if self.is_self_intersecting():
            # supplied pattern with self-intersecting panels -- it's likely to crash Maya
            # raise PatternLoadingError('{}::{}::Provided pattern has self-intersecting panels. Nothing is loaded'.format(
            #     self.__class__.__name__, self.name))
            print('{}::{}::Provided pattern has self-intersecting panels. Nothing is loaded'.format(
                self.__class__.__name__, self.name))
            self.self_inter = True
        else:
            self.self_inter = False
              
        if self.loaded_to_maya:
            print("self.name: {}, loaded_to_maya".format(self.name))
            # save the latest sim info
            self.fetchSimProps()
        
        print("load self.clean(True)")
        self.clean(True)

        # Normal flow produces garbage warnings of parenting from Maya. 
        # Solution suggestion didn't work, so I just live with them
        self.load_panels(delta_trans, parent_group)
        if rest_delay: 
            children = cmds.listRelatives(self.MayaObjects['pattern'], ad=True)
            solvers = [obj for obj in children if 'qlSolver' in obj]
            
            if len(solvers) > 0:
                solver = [ch for ch in solvers if not 'Shape' in ch][0]
                solver = cmds.rename(solver, self.name + solver)
                self.MayaObjects['solver'] = solver
                cmds.parent(solver, self.MayaObjects['pattern'])
            else:
                print("No Solver find for {}".format(self.name))
            self.loaded_to_maya = True
            return 
        else:
            if len(self.MayaObjects['panels']) > 0:
                self.stitch_panels()
                self.loaded_to_maya = True

                self.setShaderGroup(shader_group)
                self.add_colliders(obstacles)
                # self.add_attach_constrains(obstacles)
                self._setSimProps(config)
                # should be done on the mesh after stitching, res adjustment, but before sim & clean-up
                self._eval_vertex_segmentation()  
                print("load full garment")

    def load_rest(self, obstacles=[], shader_group=None, config={}):

        if len(self.MayaObjects['panels']) > 0:
            self.stitch_panels()
            self.loaded_to_maya = True

            self.setShaderGroup(shader_group)
            self.add_colliders(obstacles)
            # self.add_attach_constrains(obstacles)
            self._setSimProps(config)
            # should be done on the mesh after stitching, res adjustment, but before sim & clean-up
            self._eval_vertex_segmentation() 


    def load_panels(self, delta_trans=[0, 0, 0], parent_group=None):
        """Load panels to Maya as curve collection & geometry objects.
            Groups them by panel and by pattern"""
        # top group
        group_name = cmds.group(em=True, n=self.name)  # emrty at first

        if parent_group is not None:
            group_name = cmds.parent(group_name, parent_group)
        
        self.MayaObjects['pattern'] = group_name
        # Load panels as curves
        self.MayaObjects['panels'] = {}
        for panel_name in self.pattern['panels']:
            if 'translation' in self.pattern['panels'][panel_name]:
                self.pattern['panels'][panel_name]['translation'] = [self.pattern['panels'][panel_name]['translation'][0] + delta_trans[0], 
                                                                     self.pattern['panels'][panel_name]['translation'][1] + delta_trans[1],
                                                                     self.pattern['panels'][panel_name]['translation'][2] + delta_trans[2]]
            else:
                self.pattern['panels'][panel_name]['translation'] = [delta_trans[0], delta_trans[1], delta_trans[2]]
            panel_maya = self._load_panel(panel_name, group_name)
        # print([node for node in cmds.ls() if 'Solver' in node])
        self.loaded_to_maya = True
    
    def stitch_panels(self):
        """
            Create seams between qualoth panels.
            Assumes that panels are already loadeded (as curves).
            Assumes that after stitching every pattern becomes a single piece of geometry
            Returns
                Qulaoth cloth object name
        """
        self.MayaObjects['stitches'] = []
        for stitch in self.pattern['stitches']:
            stitch_id = qw.qlCreateSeam(
                self._maya_curve_name(stitch[0]), 
                self._maya_curve_name(stitch[1]))
            stitch_id = cmds.parent(stitch_id, self.MayaObjects['pattern'])  # organization
            self.MayaObjects['stitches'].append(stitch_id[0])

        children = cmds.listRelatives(self.MayaObjects['pattern'], ad=True)
        cloths = [obj for obj in children if 'qlCloth' in obj]
        cmds.parent(cloths, self.MayaObjects['pattern'])
        solvers = [obj for obj in children if 'qlSolver' in obj]

        if len(solvers) > 0:
            solver = [ch for ch in solvers if not 'Shape' in ch][0]
            if self.name not in solver:
                solver = cmds.rename(solver, self.name + solver)
                self.MayaObjects['solver'] = solver
                cmds.parent(solver, self.MayaObjects['pattern'])
    
    def setShaderGroup(self, shader_group=None):
        """
            Sets material properties for the cloth object created from current panel
        """

        if not self.loaded_to_maya:
            raise RuntimeError(
                'MayaGarmentError::Pattern is not yet loaded. Cannot set shader')

        if shader_group is not None:  # use previous othervise
            self.shader_group = shader_group

        if self.shader_group is not None:
            cmds.sets(self.get_qlcloth_geometry(), forceElement=self.shader_group)
    
    def save_mesh(self, folder='', tag='sim', static=False):
        """
            Saves cloth as obj file and its per vertex segmentation to a given folder or 
            to the folder with the pattern if not given.
        """
        if not self.loaded_to_maya:
            print('MayaGarmentWarning::Pattern is not yet loaded. Nothing saved')
            return

        if folder:
            filepath = folder
        else:
            filepath = self.path
        self._save_to_path(filepath, self.name + '_' + tag, static=static)
    
    def sim_caching(self, caching=True):
        """Toggles the caching of simulation steps to garment folder"""
        if caching:
            # create folder
            self.cache_path = os.path.join(self.path, self.name + '_simcache')
            try:
                os.makedirs(self.cache_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:  # ok if directory exists
                    raise
                pass
        else:
            # disable caching
            self.cache_path = '' 
    
    def  clean(self, delete=False):
        """ Hides/removes the garment from Maya scene 
            NOTE all of the maya ids assosiated with the garment become invalidated, 
            if delete flag is True
        """
        if self.loaded_to_maya:
            # Remove from simulation
            cmds.setAttr(self.get_qlcloth_props_obj() + '.active', 0)

            if delete:
                print('MayaGarment::Deleting {}'.format(self.MayaObjects['pattern']))

                 # Clean solver cache properly
                solver = qw.findSolver()
                if solver:
                    qw.qlReinitSolver(self.get_qlcloth_props_obj(), solver)

                cmds.delete(self.MayaObjects['pattern'])
                qw.deleteSolver()

                self.loaded_to_maya = False
                self.MayaObjects = {}  # clean 
            else:
                cmds.hide(self.MayaObjects['pattern'])
                try:
                    cmds.hide(self.MayaObjects['panels'])
                except Exception as e:
                    print("error")

    def display_vertex_segmentation(self):
        """
            Color every vertes of the garment according to the panel is belongs to
            (as indicated in self.vertex_labels)
        """
        # group vertices by label (it's faster then coloring one-by-one)
        vertex_select_lists = dict.fromkeys(self.panel_order() + ['stitch', 'Other'])
        for key in vertex_select_lists:
            vertex_select_lists[key] = []
        
        for vert_idx in range(len(self.current_verts)):
            str_label = self.vertex_labels[vert_idx]
            if str_label not in self.panel_order() and str_label != 'stitch':
                str_label = 'Other'

            vert_addr = '{}.vtx[{}]'.format(self.get_qlcloth_geometry(), vert_idx)
            vertex_select_lists[str_label].append(vert_addr)
        
        # Contrasting Panel Coloring for visualization
        # https://www.schemecolor.com/bright-rainbow-colors.php
        color_hex = ['FF0900', 'FF7F00', 'FFEF00', '00F11D', '0079FF', 'A800FF']
        color_list = np.empty((len(color_hex), 3))
        for idx in range(len(color_hex)):
            color_list[idx] = np.array([int(color_hex[idx][i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
        
        start_time = time.time()
        for label, str_label in enumerate(vertex_select_lists.keys()):
            if len(vertex_select_lists[str_label]) > 0: # 'Other' may not be present at all
                if str_label == 'Other':
                    color = np.ones(3)
                elif str_label == 'stitch':
                    color = np.zeros(3)
                else:
                    # color selection with expansion if the list is too small
                    factor, color_id = (label // len(color_list)) + 1, label % len(color_list)
                    color = color_list[color_id] / factor  # gets darker the more labels there are
                
                # color corresponding vertices
                cmds.select(clear=True)
                cmds.select(vertex_select_lists[str_label])
                cmds.polyColorPerVertex(rgb=color.tolist())
        
        cmds.select(clear=True)

        cmds.setAttr(self.get_qlcloth_geometry() + '.displayColors', 1)
        cmds.refresh()
    
    # ------ Simulation ------

    def _check_panel_in(self, panel_name, panels):
        if_in = False 
        for pname in panels:
            if panel_name in pname:
                if_in = True 
                break 
        return if_in
    
    def _check_attach_config(self, attach_configs, key="one_pieces"):
        name = self.name
        if isinstance(attach_configs[key], dict):
            for k, atts in attach_configs[key].items():
                if name.startswith(k):
                    return atts, 0.9 if key in ["one_pieces", "tops"] else 0.5
            ctype = self._check_predicted_bottoms()
            for k, atts in attach_configs[key].items():
                if ctype in k:
                    return atts, 0.9 if key in ["one_pieces", "tops"] else 0.5
        else:
            return attach_configs[key], 0.9 if key in ["one_pieces", "tops"] else 0.5
        raise ValueError("Cannot find attach config for {}".format(name))

    def _check_predicted_bottoms(self):
        panel_names = list(self.MayaObjects["panels"].keys())
        skirt = 0
        for panel_name in panel_names:
            if 'wb' in panel_name: return 'wb'
            elif 'pant' in panel_name: return 'pant'
            elif 'skirt' in panel_name: skirt += 1
        if skirt == 2: return 'skirt_2'
        elif skirt == 4: return 'skirt_4'
        elif skirt == 8: return 'skirt_8'

        return None


    def add_attach_constrains(self, key, attach_configs, rate=0.3, obstacles=[]):
        """
            Adds vertices to attach bodys
            Tops (include combos): vertices on the seaming line over the shoulders
            pants/skirts: the toppest veritces
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded. Cannot add attach_constrains')
        try: 
            att_inputs, attach_stiffness = self._check_attach_config(attach_configs, key)
        except Exception as e:
            print(e)

        verts_to_attach = []
        for att_int in att_inputs:
            panel_name, edge_id = att_int
            panel_node = self.MayaObjects['panels'][panel_name]['qlPattern']
            edge_node = self.MayaObjects['panels'][panel_name]['edges'][edge_id]
            if not self.name in panel_node:
                panel_node = self.name + "|" + panel_node
            verts_on_curve = qw.getVertsOnCurve(panel_node, edge_node)
            verts_to_attach += verts_on_curve
        
        points_coords = [self.current_verts[idx] for idx in verts_to_attach]
        indexs = [i[0] for i in sorted(enumerate(points_coords), key=lambda x: x[1][1], reverse=True)]
        points = []
        cloth_geo = self.get_qlcloth_geometry()
        for idx in indexs[0:int(len(indexs) * rate)]:
            points.append("{}.vtx[{}]".format(cloth_geo, verts_to_attach[idx]))
        
        new_constraints = qw.qlCreateAttachConstraint(points, obstacles)
        # set attributes
        if len(new_constraints) > 0:
            new_constraints = new_constraints[0]
            cmds.setAttr(new_constraints + '.soft', True)
            # cmds.setAttr(new_constraints + '.stiffness', self.config["constraint_stiffness"])
            cmds.setAttr(new_constraints + '.stiffness', attach_stiffness)
            new_constraints = cmds.parent(new_constraints, self.MayaObjects['pattern'])

            if 'constraints' not in self.MayaObjects:
                self.MayaObjects['constraints'] = []
            self.MayaObjects['constraints'].append(new_constraints)
    
    def add_colliders(self, obstacles=[]):
        """
            Adds given Maya objects as colliders of the garment
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded. Cannot load colliders')
        if obstacles: # if not given, use previous ones
            self.obstacles = obstacles
        
        if 'colliders' not in self.MayaObjects:
            self.MayaObjects['colliders'] = []
        
        for obj in self.obstacles:
            collider = qw.qlCreateCollider(self.get_qlcloth_geometry(), obj)
            # apply current friction settings
            qw.setColliderFriction(collider, self.config['body_friction'])
            # organize object tree
            collider = cmds.parent(collider, self.MayaObjects['pattern'])
            self.MayaObjects['colliders'].append(collider)
    
    def fetchSimProps(self):
        """Fetch garment material & body friction from Maya settings"""
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded.')
        
        self.config['material'] = qw.fetchFabricProps(self.get_qlcloth_props_obj())  
        if 'colliders' in self.MayaObjects and self.MayaObjects['colliders']:
            # assuming all colliders have the same value
            friction = qw.fetchColliderFriction(self.MayaObjects['colliders'][0])  
            if friction:
                self.config['body_friction'] = friction
        self.config['collision_thickness'] = cmds.getAttr(self.get_qlcloth_props_obj() + '.thickness')
        # take resolution scale from any of the panels assuming all the same
        self.config['resolution_scale'] = qw.fetchPanelResolution()
        return self.config
    
    def update_verts_info(self):
        """
            Retrieves current vertex positions from Maya & updates the last state.
            For best performance, should be called on each iteration of simulation
            Assumes the object is already loaded & stitched
        """
        # working with meshes http://www.fevrierdorian.com/blog/post/2011/09/27/Quickly-retrieve-vertex-positions-of-a-Maya-mesh-%28English-Translation%29
        cloth_dag = self.get_qlcloth_geom_dag()
        mesh = OpenMaya.MFnMesh(cloth_dag)
        vertices = utils.get_vertices_np(mesh)
        self.last_verts = self.current_verts
        self.current_verts = vertices
    
    def update_verts_info2(self):
        """
            Don't know why, need to be called after every frame simulation
        """
        cloth_dag = self.get_qlcloth_geom_dag()
        mesh = OpenMaya.MFnMesh(cloth_dag)
    
    def cache_if_enabled(self, frame):
        """If caching is enabled -> saves current geometry to cache folder
            Does nothing otherwise """
        if not self.loaded_to_maya:
            print('MayaGarmentWarning::Pattern is not yet loaded. Nothing cached')
            return

        if hasattr(self, 'cache_path') and self.cache_path:
            self._save_to_path(self.cache_path, self.name + '_{:04d}'.format(frame))
    
    # ------ Qualoth objects ------
    def get_qlcloth_geometry(self):
        """
            Find the first Qualoth cloth geometry object belonging to current pattern
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded.')

        if 'qlClothOut' not in self.MayaObjects:
            children = cmds.listRelatives(self.MayaObjects['pattern'], ad=True)
            cloths = [obj for obj in children 
                      if 'qlCloth' in obj and 'Out' in obj and 'Shape' not in obj]
            self.MayaObjects['qlClothOut'] = cloths[0]
        
        return self.MayaObjects['qlClothOut']
    
    def get_qlcloth_props_obj(self):
        """
            Find the first qlCloth object belonging to current pattern
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded.')
        
        if 'qlCloth' not in self.MayaObjects:
            children = cmds.listRelatives(self.MayaObjects['pattern'], ad=True)
            cloths = [obj for obj in children 
                      if 'qlCloth' in obj and 'Out' not in obj and 'Shape' in obj]
            self.MayaObjects['qlCloth'] = cloths[0]
        
        return self.MayaObjects['qlCloth']
    
    def get_qlcloth_geom_dag(self):
        """
            returns DAG reference to cloth shape object
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded.')
        
        if 'shapeDAG' not in self.MayaObjects:
            self.MayaObjects['shapeDAG'] = utils.get_dag(self.get_qlcloth_geometry())
        return self.MayaObjects['shapeDAG']

    # ------ Geometry Checks ------
    def is_static(self, threshold, allowed_non_static_percent=0):
        """
            Checks wether garment is in the static equilibrium
            Compares current state with the last recorded state
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded. Cannot check static')
        
        if self.last_verts is None:  # first iteration
            return False 
        
        # Compare L1 norm per vertex
        # Checking vertices change is the same as checking if velocity is zero
        diff = np.abs(self.current_verts - self.last_verts)
        diff_L1 = np.sum(diff, axis=1)

        non_static_len = len(diff_L1[diff_L1 > threshold])  # compare vertex-wize to allow accurate control over outliers

        if non_static_len == 0 or non_static_len < len(self.current_verts) * 0.01 * allowed_non_static_percent:  
            print('\nStatic with {} non-static vertices out of {}'.format(non_static_len, len(self.current_verts)))
            return True, non_static_len
        else:
            return False, non_static_len
    
    def intersect_colliders_3D(self, obstacles=[], static=False, logger=None):
        """Checks wheter garment intersects given obstacles or its colliders if obstacles are not given
            Returns True if intersections found

            Having intersections may disrupt simulation result although it seems to recover from some of those
        """
        if not self.loaded_to_maya:
            raise RuntimeError('Garment is not yet loaded: cannot check for intersections')
        
        if not obstacles:
            obstacles = self.obstacles
        
        # check intersection with colliders (max 2, body and floor)
        num_obs = 2 if not static else len(obstacles)
        for obj in obstacles[:num_obs]:
            intersecting = self._intersect_object(obj, logger=logger)

            if intersecting:
                return True
        return False
    
    def self_intersect_3D(self, verbose=False, logger=None):
        """Checks wheter currently loaded garment geometry intersects itself
        Unline boolOp, check is non-invasive and do not require garment reload or copy.
        
        Having intersections may disrupt simulation result although it seems to recover from some of those
        """
        if not self.loaded_to_maya:
            raise RuntimeError(
                'MayaGarmentError::Pattern is not yet loaded. Cannot check geometry self-intersection')
        
        # It turns out that OpenMaya python reference has nothing to do with reality of passing argument:
        # most of the functions I use below are to be treated as wrappers of c++ API
        # https://help.autodesk.com/view/MAYAUL/2018//ENU/?guid=__cpp_ref_class_m_fn_mesh_html
        mesh, cloth_dag = utils.get_mesh_dag(self.get_qlcloth_geometry())

        vertices = OpenMaya.MPointArray()
        mesh.getPoints(vertices, OpenMaya.MSpace.kWorld)

        # use ray intersect with all edges of current mesh & the mesh itself
        num_edges = mesh.numEdges()
        accelerator = mesh.autoUniformGridParams()
        num_hits = 0
        for edge_id in range(num_edges):
            # Vertices that comprise an edge
            vtx1, vtx2 = utils.edge_vert_ids(mesh, edge_id)
            # test intersection
            raySource = OpenMaya.MFloatPoint(vertices[vtx1])
            rayDir = OpenMaya.MFloatVector(vertices[vtx2] - vertices[vtx1])
            hit, hitFaces, hitPoints, _ = utils.test_ray_intersect(mesh, raySource, rayDir, accelerator, return_info=True)
            if not hit:
                continue
            # Since edge is on the mesh, we have tons of false hits
            # => check if current edge is adjusent to hit faces: if shares a vertex
            for face_id in range(hitFaces.length()):
                face_verts = OpenMaya.MIntArray()
                mesh.getPolygonVertices(hitFaces[face_id], face_verts)
                face_verts = [face_verts[j] for j in range(face_verts.length())]
                if vtx1 not in face_verts and vtx2 not in face_verts:
                    # hit face is not adjacent to the edge => real hit
                    if verbose:
                        print('Hit point: {}, {}, {}'.format(hitPoints[face_id][0], hitPoints[face_id][1], hitPoints[face_id][2]))
                    num_hits += 1

        if num_hits == 0:  # no intersections -- no need for threshold check
            print('{} is not self-intersecting'.format(self.name))
            if logger is not None:
                logger.info('{} is not self-intersecting'.format(self.name))
            return False
        
        if ('self_intersect_hit_threshold' in self.config 
                and num_hits > self.config['self_intersect_hit_threshold']
                or num_hits > 0 and 'self_intersect_hit_threshold' not in self.config):  # non-zero hit if no threshold provided
            
            if logger is not None:
                logger.info('{} is self-intersecting with {} intersect edges -- above threshold {}'.format(
                    self.name, num_hits,
                    self.config['self_intersect_hit_threshold'] if 'self_intersect_hit_threshold' in self.config else 0))
            else:
                print('{} is self-intersecting with {} intersect edges -- above threshold {}'.format(
                    self.name, num_hits,
                    self.config['self_intersect_hit_threshold'] if 'self_intersect_hit_threshold' in self.config else 0))
            return True
        else:
            # no need to reload -- non-invasive checks 
            if logger is not None:
                logger.info('{} is self-intersecting with {} intersect edges -- ignored by threshold {}'.format(
                    self.name, num_hits,
                    self.config['self_intersect_hit_threshold'] if 'self_intersect_hit_threshold' in self.config else 0))
            else:
                print('{} is self-intersecting with {} intersect edges -- ignored by threshold {}'.format(
                    self.name, num_hits,
                    self.config['self_intersect_hit_threshold'] if 'self_intersect_hit_threshold' in self.config else 0))

            return False
    
    def set_sim_props(self, shader_name=None, sim_name=None, sim_path=None):
        if shader_name is None and sim_name is None:
            print("Keep default, no update")
        elif sim_name is not None and os.path.exists(sim_name):
            with open(sim_name, 'r') as f:
                sim_props = json.load(f)
            self._setSimProps(config=sim_props)
            shader_name = os.path.basename(sim_name).split(".")[0]
        else:
            if os.path.exists(os.path.join(sim_path, shader_name + ".json")):
                sim_name = os.path.join(sim_path, shader_name + ".json")
                with open(sim_name, 'r') as f:
                    sim_props = json.load(f)
                self._setSimProps(config=sim_props)
        return shader_name, sim_name
    
    # ------ ~Private -------
    def _load_panel(self, panel_name, pattern_group=None):
        """
            Loads curves contituting given panel to Maya. 
            Goups them per panel
        """
        panel = self.pattern['panels'][panel_name]

        vertices = np.asarray(panel['vertices'])
        self.MayaObjects['panels'][panel_name] = {}
        self.MayaObjects['panels'][panel_name]['edges'] = []

        # top panel group
        panel_group = cmds.group(n=panel_name, em=True)
        if pattern_group is not None:
            panel_group = cmds.parent(panel_group, pattern_group)[0]
        self.MayaObjects['panels'][panel_name]['group'] = panel_group

        # draw edges
        curve_names = []
        for edge in panel['edges']:
            curve_points = self._edge_as_3d_tuple_list(edge, vertices)
            curve = cmds.curve(p=curve_points, d=(len(curve_points) - 1))
            curve_names.append(curve)
            self.MayaObjects['panels'][panel_name]['edges'].append(curve)
        # Group  
        curve_group = cmds.group(curve_names, n=panel_name + '_curves')
        curve_group = cmds.parent(curve_group, panel_group)[0]
        self.MayaObjects['panels'][panel_name]['curve_group'] = curve_group
        # 3D placemement
        self._apply_panel_3d_placement(panel_name)

        # Create geometry
        panel_geom = qw.qlCreatePattern(curve_group)
        # take out the solver node -- created only once per scene, no need to store
        # solvers = [obj for obj in panel_geom if 'Solver' in obj]
        # print("solvers", solvers)
        # panel_geom = list(set(panel_geom) - set(solvers))
        panel_geom = cmds.parent(panel_geom, panel_group)  # organize

        pattern_object = [node for node in panel_geom if 'Pattern' in node]
        
        self.MayaObjects['panels'][panel_name]['qlPattern'] = (
            pattern_object[0] if panel_group in pattern_object[0] else panel_group + '|' + pattern_object[0]
        )

        return panel_group

    def _setSimProps(self, config={}):
        """Pass material properties for cloth & colliders to Qualoth"""
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded.')
        
        if config:
            self.config = config
        
        qw.setFabricProps( self.get_qlcloth_props_obj(),  self.config['material'])


        if 'colliders' in self.MayaObjects:
            for collider in self.MayaObjects['colliders']:
                qw.setColliderFriction(collider, self.config['body_friction'])

        if 'collision_thickness' in self.config:
            # if not provided, use default auto-calculated value
            cmds.setAttr(self.get_qlcloth_props_obj() + '.overrideThickness', 1)
            cmds.setAttr(self.get_qlcloth_props_obj() + '.thickness', self.config['collision_thickness'])
        
        # if 'selfCollision' in self.config:
        #     cmds.setAttr(self.get_qlcloth_props_obj() + '.selfCollision', 1)


        # update resolution properties
        qw.setPanelsResolution(self.config['resolution_scale'])
    
    def _eval_vertex_segmentation(self):
        """
            Evalute which vertex belongs to which panel
            NOTE: only applicable to the mesh that was JUST loaded and stitched
                -- Before the mesh was cleaned up (because the info from Qualoth is dependent on original topology) 
                -- before the sim started (need planarity checks)
                Hence fuction is only called once on garment load
            NOTE: if garment resolution was changed from Maya tools, 
                the segmentation is not guranteed to be consistent with the change, 
                (reload garment geometry to get correct segmentation)
        """

        if not self.loaded_to_maya:
            raise RuntimeError('Garment should be loaded when evaluating vertex segmentation')
        
        self.update_verts_info()
        self.vertex_labels = [None] * len(self.current_verts)

        # -- Stitches (provided in qualoth objects directly) ---
        on_stitches = self._verts_on_stitches()  # TODO I can even distinguish stitches from each other!
        for idx in on_stitches:
            self.vertex_labels[idx] = 'stitch'
        
        # --- vertices ---
        vertices = self.current_verts
        # BBoxes give fast results for most vertices
        bboxes = self._all_panel_bboxes() 
        vertices_multi_match = []
        for i in range(len(vertices)):
            if i in on_stitches:  # already labeled
                continue
            vertex = vertices[i]
            # check which panel is the closest one
            in_bboxes = []
            for panel in bboxes:
                if self._point_in_bbox(vertex, bboxes[panel]):
                    in_bboxes.append(panel)
            
            if len(in_bboxes) == 1:
                self.vertex_labels[i] = in_bboxes[0]
            else:  # multiple or zero matches -- handle later
                vertices_multi_match.append((i, in_bboxes))
        

        # eval for confusing cases
        neighbour_checks = 0
        while len(vertices_multi_match) > 0:
            unlabeled_vert_id, matched_panels = vertices_multi_match.pop(0)

            # check if vert in on the plane of any of the panels
            on_panel_planes = []
            for panel in matched_panels:
                if self._point_on_plane(vertices[unlabeled_vert_id], panel):
                    on_panel_planes.append(panel)

            # plane might not be the only option 
            if len(on_panel_planes) == 1:  # found!
                self.vertex_labels[unlabeled_vert_id] = on_panel_planes[0]
            else:
                # by this time, many vertices already have labels, so let's just borrow from neigbours
                neighbors = self._get_vert_neighbours(unlabeled_vert_id)

                neighbour_checks += 1

                if len(neighbors) == 0:
                    # print('Skipped Vertex {} with zero neigbors'.format(unlabeled_vert_id))
                    continue

                unlabelled = [unl[0] for unl in vertices_multi_match]
                # check only labeled neigbors that are not on stitches
                neighbors = [vert_id for vert_id in neighbors if vert_id not in unlabelled and vert_id not in on_stitches]

                if len(neighbors) > 0:
                    neighbour_labels = [self.vertex_labels[vert_id] for vert_id in neighbors]
                    
                    # https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list
                    frequent_label = max(set(neighbour_labels), key=neighbour_labels.count)
                    self.vertex_labels[unlabeled_vert_id] = frequent_label
                else:
                    # put back 
                    # NOTE! There is a ponetial for infinite loop here, but it shoulf not occur
                    # if the garment is freshly loaded before sim
                    print('Garment::Labelling::vertex {} needs revisit'.format(unlabeled_vert_id))
                    vertices_multi_match.append((unlabeled_vert_id, on_panel_planes))
    
    def _clean_mesh(self):
        """
        Clean mesh from incosistencies introduces by stitching, 
        and update vertex-dependednt info accordingly
        """
        # remove the junk after garment was stitched and labeled
        cmds.polyClean(self.get_qlcloth_geometry())

        # fix labeling
        self.update_verts_info()
        match_verts = utils.match_vert_lists(self.current_verts, self.last_verts)

        self.vertex_labels = [self.vertex_labels[i] for i in match_verts]
    
    def _edge_as_3d_tuple_list(self, edge, vertices):
        """
            Represents given edge object as list of control points
            suitable for draing in Maya
        """
        points = vertices[edge['endpoints'], :]
        if 'curvature' in edge:
            control_coords = self._control_to_abs_coord(
                points[0], points[1], edge['curvature']
            )
            # Rearrange
            points = np.r_[
                [points[0]], [control_coords], [points[1]]
            ]
        # to 3D
        points = np.c_[points, np.zeros(len(points))]

        return list(map(tuple, points))

    def _applyEuler(self, vector, eulerRot):
        """Applies Euler angles (in degrees) to provided 3D vector"""
        # https://www.cs.utexas.edu/~theshark/courses/cs354/lectures/cs354-14.pdf
        eulerRot_rad = np.deg2rad(eulerRot)
        # X 
        vector_x = np.copy(vector)
        vector_x[1] = vector[1] * np.cos(eulerRot_rad[0]) - vector[2] * np.sin(eulerRot_rad[0])
        vector_x[2] = vector[1] * np.sin(eulerRot_rad[0]) + vector[2] * np.cos(eulerRot_rad[0])

        # Y
        vector_y = np.copy(vector_x)
        vector_y[0] = vector_x[0] * np.cos(eulerRot_rad[1]) + vector_x[2] * np.sin(eulerRot_rad[1])
        vector_y[2] = -vector_x[0] * np.sin(eulerRot_rad[1]) + vector_x[2] * np.cos(eulerRot_rad[1])

        # Z
        vector_z = np.copy(vector_y)
        vector_z[0] = vector_y[0] * np.cos(eulerRot_rad[2]) - vector_y[1] * np.sin(eulerRot_rad[2])
        vector_z[1] = vector_y[0] * np.sin(eulerRot_rad[2]) + vector_y[1] * np.cos(eulerRot_rad[2])

        return vector_z
    
    def _set_panel_3D_attr(self, panel_dict, panel_group, attribute, maya_attr):
        """Set recuested attribute to value from the spec"""
        if attribute in panel_dict:
            values = panel_dict[attribute]
        else:
            values = [0, 0, 0]
        
        cmds.setAttr(panel_group + '.' + maya_attr, values[0], values[1], values[2], type='double3')
    
    def _apply_panel_3d_placement(self, panel_name):
        """Apply transform from spec to given panel"""
        panel = self.pattern['panels'][panel_name]
        panel_group = self.MayaObjects['panels'][panel_name]['curve_group']

        # set pivot to origin relative to currently loaded curves
        cmds.xform(panel_group, pivots=[0, 0, 0], worldSpace=True)

        # now place correctly
        self._set_panel_3D_attr(panel, panel_group, 'translation', 'translate')
        self._set_panel_3D_attr(panel, panel_group, 'rotation', 'rotate')
    
    def _maya_curve_name(self, address):
        """ Shortcut to retrieve the name of curve corresponding to the edge"""
        panel_name = address['panel']
        edge_id = address['edge']
        return self.MayaObjects['panels'][panel_name]['edges'][edge_id]
   
    def _save_to_path(self, path, filename, static=False):
        """Save current state of cloth object to given path with given filename"""
        
        if static:
            # update UV map
            cmds.select(self.get_qlcloth_geometry())
            cmds.polyMultiLayoutUV(layoutMethod=1, layout=2, scale=1)
            uv_filepath = os.path.join(path, filename + '_uv.png')
            cmds.uvSnapshot(o=True, name=uv_filepath, aa=True, xResolution=1024, yResolution=1024, ff='png', r=255, g=255, b=255)

        # geometry
        filepath = os.path.join(path, filename + '.obj')
        utils.save_mesh(self.get_qlcloth_geometry(), filepath)

        if static:
            # segmentation
            filepath = os.path.join(path, filename + '_segmentation.txt')
            with open(filepath, 'w') as f:
                for panel_name in self.vertex_labels:
                    f.write("%s\n" % panel_name)
        
        # eval
        num_verts = cmds.polyEvaluate(self.get_qlcloth_geometry(), v=True)
        if num_verts != len(self.vertex_labels):
            print('MayaGarment::WARNING::Segmentation list does not match mesh topology in save {}'.format(self.name))
    
    def _intersect_object(self, geometry, logger=None):
        """Check if given object intersects current cloth geometry
            Function does not have side-effects on input geometry"""
        
        # ray-based intersection test
        cloth_mesh, cloth_dag = utils.get_mesh_dag(self.get_qlcloth_geometry())
        obstacle_mesh, obstacle_dag = utils.get_mesh_dag(geometry)

        # use obstacle verts as a base for testing
        # Assuming that the obstacle geometry has a lower resolution then the garment
        obs_vertices = OpenMaya.MPointArray()
        obstacle_mesh.getPoints(obs_vertices, OpenMaya.MSpace.kWorld)

        # use ray intersect of all edges of obstacle mesh with the garment mesh
        num_edges = obstacle_mesh.numEdges()
        accelerator = cloth_mesh.autoUniformGridParams()
        hit_border_length = 0  # those are edges along the border of intersecting area on the geometry
        for edge_id in range(num_edges):
            # Vertices that comprise an edge
            vtx1, vtx2 = utils.edge_vert_ids(obstacle_mesh, edge_id)
            # test intersection
            raySource = OpenMaya.MFloatPoint(obs_vertices[vtx1])
            rayDir = OpenMaya.MFloatVector(obs_vertices[vtx2] - obs_vertices[vtx1])
            hit = utils.test_ray_intersect(cloth_mesh, raySource, rayDir, accelerator)
            if hit: 
                # A very naive approximation of total border length of areas of intersection
                hit_border_length += rayDir.length()
        
        if hit_border_length < 1e-5:  # no intersections -- no need for threshold check
            print('{} with {} do not intersect'.format(geometry, self.name))
            return False
        
        if ('object_intersect_border_threshold' in self.config 
                and hit_border_length > 1000
                or (hit_border_length > 1e-5 and 'object_intersect_border_threshold' not in self.config)):  # non-zero hit if no threshold provided
            print('{} with {} intersect::Approximate intersection border length {:.2f} cm is above threshold {:.2f} cm'.format(
                geometry, self.name, hit_border_length, 
                self.config['object_intersect_border_threshold'] if 'object_intersect_border_threshold' in self.config else 0))
            if logger is not None:
                logger.info('{} with {} intersect::Approximate intersection border length {:.2f} cm is above threshold {:.2f} cm'.format(
                    geometry, self.name, hit_border_length, 
                    self.config['object_intersect_border_threshold'] if 'object_intersect_border_threshold' in self.config else 0))
            return True
        
        print('{} with {} intersect::Approximate intersection border length {:.2f} cm is ignored by threshold {:.2f} cm'.format(
            geometry, self.name, hit_border_length, 
            self.config['object_intersect_border_threshold'] if 'object_intersect_border_threshold' in self.config else 0))
        if logger is not None:
            logger.info('{} with {} intersect::Approximate intersection border length {:.2f} cm is ignored by threshold {:.2f} cm'.format(
                geometry, self.name, hit_border_length, 
                self.config['object_intersect_border_threshold'] if 'object_intersect_border_threshold' in self.config else 0))
        return False
    
    def _verts_on_stitches(self):
        """
            List all the vertices in garment mesh located on stitches
            NOTE: it does not output vertices correctly on is the mesh topology was changed 
                (e.g. after cmds.polyClean())!!
        """
        on_stitches = []
        for stitch in self.pattern['stitches']:
            # querying one side is enough since they share the vertices
            for side in [0, 1]:
                stitch_curve = self._maya_curve_name(stitch[side]) 
                panel_name = stitch[side]['panel']
                panel_node = self.MayaObjects['panels'][panel_name]['qlPattern']

                verts_on_curve = qw.getVertsOnCurve(panel_node, stitch_curve)
 
                on_stitches += verts_on_curve
        return on_stitches

    def _verts_on_curves(self):
        """
            List all the vertices of garment mesh that are located on panel curves
        """
        all_edge_verts = []
        for panel in self.panel_order():
            panel_info = self.MayaObjects['panels'][panel]
            panel_node = panel_info['qlPattern']

            # curves
            for curve in panel_info['edges']:
                verts_on_curve = qw.getVertsOnCurve(panel_node, curve)
                all_edge_verts += verts_on_curve
    
                print(min(verts_on_curve), max(verts_on_curve))

        # might contain duplicates
        return all_edge_verts
    
    def _all_panel_bboxes(self):
        """
            Evaluate 3D bounding boxes for all panels (as original curve loops)
        """
        panel_curves = self.MayaObjects['panels']
        bboxes = {}
        for panel in panel_curves:
            box = cmds.exactWorldBoundingBox(panel_curves[panel]['curve_group'])
            bboxes[panel] = box
        return bboxes
    
    @staticmethod
    def _point_in_bbox(point, bbox, tol=0.01):
        """
            Check if point is within bbox
            bbbox given in maya format (float[]	xmin, ymin, zmin, xmax, ymax, zmax.)
            NOTE: tol value is needed for cases when BBox collapses to 2D
        """
        if (point[0] < (bbox[0] - tol) or point[0] > (bbox[3] + tol)
                or point[1] < (bbox[1] - tol) or point[1] > (bbox[4] + tol)
                or point[2] < (bbox[2] - tol) or point[2] > (bbox[5] + tol)):
            return False
        return True
    
    def _point_on_plane(self, point, panel, tol=0.001):
        """
            Check if a point belongs to the same plane as given in the curve group
        """
        # I could check by panel rotation and translation!!
        rot = self.pattern['panels'][panel]['rotation']
        transl = np.array(self.pattern['panels'][panel]['translation'])

        # default panel normal upon load, sign doesn't matter here
        normal = np.array([0., 0., 1.])  
        rotated_normal = self._applyEuler(normal, rot)

        dot_prod = np.dot(np.array(point) - transl, rotated_normal)

        return np.isclose(dot_prod, 0., atol=tol)
    
    def _get_vert_neighbours(self, vert_id):
        """
            List the neigbours of given vertex in current cloth mesh
        """
        mesh_name = self.get_qlcloth_geometry()

        edges = cmds.polyListComponentConversion(
            mesh_name + '.vtx[%d]' % vert_id, 
            fromVertex=True, toEdge=True)

        neighbors = []
        for edge in edges:
            neighbor_verts_str = cmds.polyListComponentConversion(edge, toVertex=True)
            for neighbor_str in neighbor_verts_str:
                values = neighbor_str.split(']')[0].split('[')[-1]
                if ':' in values:
                    neighbors += [int(x) for x in values.split(':')]
                else:
                    neighbors.append(int(values))
        
        return list(set(neighbors))  # leave only unique
    
    def _panel_to_id(self, panel):
        """ 
            Panel label as integer given the name of the panel
        """
        return int(self.panel_order().index(panel) + 1)


            
        

        