"""Shares utils to work with Maya"""

import ctypes
import os
import numpy as np
import math
from maya import OpenMaya
from maya import cmds
from json import JSONEncoder
import json


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# ----- Working with files -----
def load_file(filepath, name='object'):
    """Load mesh to the scene"""
    if not os.path.isfile(filepath):
        raise RuntimeError('Loading Object from file to Maya::Missing file {}'.format(filepath))

    obj = cmds.file(filepath, i=True, rnn=True)[0]
    obj = cmds.rename(obj, name + '#')

    return obj


def save_mesh(target, to_file):
    """Save given object to file as a mesh"""

    # Make sure to only select requested mesh
    cmds.select(clear=True)
    cmds.select(target)

    cmds.file(
        to_file,
        type='OBJExport',  
        exportSelectedStrict=True,  # export selected -- only explicitely selected
        options='groups=0;ptgroups=0;materials=0;smoothing=0;normals=1',  # very simple obj
        force=True,   # force override if file exists
        defaultExtensions=False
    )

    cmds.select(clear=True)


# ----- Mesh info -----
def get_dag(object_name):
    """Return DAG for requested object"""
    selectionList = OpenMaya.MSelectionList()
    selectionList.add(object_name)
    dag = OpenMaya.MDagPath()
    selectionList.getDagPath(0, dag)
    return dag


def get_mesh_dag(object_name):
    """Return MFnMesh object by the object name"""
    # get object as OpenMaya object -- though DAG
    dag = get_dag(object_name)
    # as mesh
    mesh = OpenMaya.MFnMesh(dag)  # reference https://help.autodesk.com/view/MAYAUL/2017/ENU/?guid=__py_ref_class_open_maya_1_1_m_fn_mesh_html

    return mesh, dag


def get_vertices_np(mesh):
    """
        Retreive vertex info as np array for given mesh object
    """
    maya_vertices = OpenMaya.MPointArray()
    mesh.getPoints(maya_vertices, OpenMaya.MSpace.kWorld)

    vertices = np.empty((maya_vertices.length(), 3))
    for i in range(maya_vertices.length()):
        for j in range(3):
            vertices[i, j] = maya_vertices[i][j]

    return vertices


def match_vert_lists(short_list, long_list):
    """
        Find the vertices from long list that correspond to verts in short_list
        Both lists are numpy arrays
        NOTE: Assuming order is matching => O(len(long_list)) complexity: 
            order of vertices in short list is the same as in long list (for those that are left)
    """
    match_list = []

    idx_short = 0
    for idx_long in range(len(long_list)):
        long_vertex = long_list[idx_long]
        short_vertex = short_list[idx_short]

        if all(np.isclose(short_vertex, long_vertex, atol=1e-5)):
            match_list.append(idx_long)
            idx_short += 1  # advance the short list indexing
            if idx_short >= len(short_list):  # short list finished before the long one
                break
    
    if len(match_list) != len(short_list):
        raise ValueError('Vertex matching unsuccessfull: matched {} of {} vertices in short list'.format(
            len(match_list), len(short_list)
        ))
    
    return match_list


# ---- Mesh operations ----
def test_ray_intersect(mesh, raySource, rayVector, accelerator=None, hit_tol=None, return_info=False):
    """Check if given ray intersect given mesh
        * hit_tol ignores intersections that are within hit_tol from the ray source (as % of ray length) -- usefull when checking self-intersect
        * mesh is expected to be of MFnMesh type
        * accelrator is a stucture for speeding-up calculations.
            It can be initialized from MFnMesh object and should be supplied with every call to this function
    """
     # It turns out that OpenMaya python reference has nothing to do with reality of passing argument:
    # most of the functions I use below are to be treated as wrappers of c++ API
    # https://help.autodesk.com/view/MAYAUL/2018//ENU/?guid=__cpp_ref_class_m_fn_mesh_html

    # follow structure https://stackoverflow.com/questions/58390664/how-to-fix-typeerror-in-method-mfnmesh-anyintersection-argument-4-of-type
    maxParam = 1  # only search for intersections within given vector
    testBothDirections = False  # only in the given direction
    sortHits = False  # no need to waste time on sorting

    hitPoints = OpenMaya.MFloatPointArray()
    hitRayParams = OpenMaya.MFloatArray()
    hitFaces = OpenMaya.MIntArray()
    hit = mesh.allIntersections(
        raySource, rayVector, None, None, False, OpenMaya.MSpace.kWorld, maxParam, testBothDirections, accelerator, sortHits,
        hitPoints, hitRayParams, hitFaces, None, None, None, 1e-6)   # TODO anyIntersection
    
    if hit and hit_tol is not None:
        hit = any([dist > hit_tol for dist in hitRayParams])

    if return_info:
        return hit, hitFaces, hitPoints, hitRayParams
    
    return hit


def edge_vert_ids(mesh, edge_id):
    """Return vertex ids for a given edge in given mesh"""
    # Have to go through the C++ wrappers
    # Vertices that comprise an edge
    script_util = OpenMaya.MScriptUtil(0.0)
    v_ids_cptr = script_util.asInt2Ptr()  # https://forums.cgsociety.org/t/mfnmesh-getedgevertices-error-on-2011/1652362
    mesh.getEdgeVertices(edge_id, v_ids_cptr) 

    # get values from SWIG pointer https://stackoverflow.com/questions/39344039/python-cast-swigpythonobject-to-python-object
    ty = ctypes.c_uint * 2
    v_ids_list = ty.from_address(int(v_ids_cptr))
    return v_ids_list[0], v_ids_list[1]


def scale_to_cm(target, max_height_cm=220):
    """Heuristically check the target units and scale to cantimeters if other units are detected
        * default value of max_height_cm is for meshes of humans
    """
    # check for througth height (Y axis)
    # NOTE prone to fails if non-meter units are used for body
    bb = cmds.polyEvaluate(target, boundingBox=True)  # ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    height = bb[1][1] - bb[1][0]
    if height < max_height_cm * 0.01:  # meters
        cmds.scale(100, 100, 100, target, centerPivot=True, absolute=True)
        print('Warning: {} is found to use meters as units. Scaled up by 100 for cm'.format(target))
    elif height < max_height_cm * 0.01:  # decimeters
        cmds.scale(10, 10, 10, target, centerPivot=True, absolute=True)
        print('Warning: {} is found to use decimeters as units. Scaled up by 10 for cm'.format(target))
    elif height > max_height_cm:  # millimiters or something strange
        cmds.scale(0.1, 0.1, 0.1, target, centerPivot=True, absolute=True)
        print('Warning: {} is found to use millimiters as units. Scaled down by 0.1 for cm'.format(target))


def eulerAngleToRoatationMatrix(theta):
    R_x = np.array([[1,                   0,                   0],
                    [0,                   math.cos(theta[0]),  -math.sin(theta[0])],
                    [0,                   math.sin(theta[0]),  math.cos(theta[0])]
                    ])
    R_y = np.array([[ math.cos(theta[1]), 0,                   math.sin(theta[1])],
                    [0,                   1,                   0],
                    [-math.sin(theta[1]), 0,                   math.cos(theta[1])]
                    ])
    R_z = np.array([[ math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [ math.sin(theta[2]),  math.cos(theta[2]), 0],
                    [ 0,                  0,                   1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return math.degrees(x), math.degrees(y), math.degrees(z)


#---- pose operations ----
def load_pose_data(data_file):
    spin = True if data_file.endswith(".json") else False
    if spin:
        data = json.load(open(data_file, "r"))
        if "rotmat_tuned" in data:
            rotmat = np.array(data["rotmat_tuned"])
        else:
            rotmat = np.array(data["rotmat"])
        poses = []
        num_bones = rotmat.shape[0] // (3*3)
        for i in range(0, rotmat.shape[0], 9):
            mat = rotmat[i:i+9].reshape(3, 3)
            if i == 0:
                extr_rotmat = eulerAngleToRoatationMatrix([math.radians(180), math.radians(0), math.radians(0)])
                mat = np.dot(extr_rotmat, mat)
            x, y, z = rotationMatrixToEulerAngles(mat)
            poses.extend([x, y, z])
        poses = np.array(poses).reshape(1, -1)
        trans = np.zeros((1, 3))
        # trans[0, 1] = 90
        trans[0, 2] = 0.85
        total_frames = poses.shape[0]
        return poses, trans, total_frames, ""

    else:
        data = np.load(data_file)
        if 'poses' in data.keys():
            poses = data['poses']
            N = poses.shape[0]
            cdata_ids = list(range(int(0.1*N), int(0.9*N),1))
            poses = data['poses'][cdata_ids].astype(np.float32)
            trans = data['trans'][cdata_ids].astype(np.float32)
            total_frames = poses.shape[0]
            gender = data['gender']
            return poses, trans, total_frames, str(gender.astype('<U13'))
    return None, None, 0, None

