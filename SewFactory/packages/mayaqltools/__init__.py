"""
    Package for to simulate garments from patterns in Maya with Qualoth
    Note that Maya uses Python 2.7 (incl Maya 2020) hence this package is adapted to Python 2.7

    Main dependencies:
        * Maya 2018+
        * Arnold Renderer
        * Qualoth (compatible with your Maya version)
    
    To run the package in Maya don't foget to add it to PYTHONPATH!
"""
#import mayascene
#reload(mayascene)

#from .mayascene import PatternLoadingError, MayaGarment, Scene, MayaGarmentWithUI, MySceneUI

from imp import reload

import simulation
import qualothwrapper
import garment_objs
import fbx_animation
import maya_scene
import materials
import playblast

import utils
import garment


reload(simulation)
reload(qualothwrapper)
reload(garment_objs)
reload(fbx_animation)
reload(maya_scene)
reload(materials)
reload(playblast)

reload(utils)
reload(garment)
