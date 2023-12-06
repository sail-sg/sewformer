import numpy as np
import torch
import torch.nn as nn

# from smplx.lbs import lbs
from data.human_body_prior.lbs import lbs
import sys

class BodyModel(nn.Module):
    
    def __init__(self,
                 bm_fname, num_betas=10, num_dmpls=None, dmpl_fname=None, num_expressions=80,
                 use_posedirs=True, dtype=torch.float32, persistant_buffer=False): 
        super(BodyModel, self).__init__()

        self.dtype = dtype
        # -- Load SMPL params --
        if '.npz' in bm_fname:
            smpl_dict = np.load(bm_fname, encoding='latin1')
        else:
            raise ValueError('bm_fname should be either a .pkl nor .npz file')

        # these are supposed for later convenient look up
        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        self.num_expressions = num_expressions

        njoints = smpl_dict['posedirs'].shape[2] // 3
        self.model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano', 105: 'animal_horse', 102: 'animal_dog', }[njoints]

        self.use_dmpl = False
        if num_dmpls is not None:
            if dmpl_fname is not None:
                self.use_dmpl = True
            else:
                raise (ValueError('dmpl_fname should be provided when using dmpls!'))
        
        if self.use_dmpl and self.model_type in ['smplx', 'mano', 'animal_horse', 'animal_dog']: 
            raise (NotImplementedError('DMPLs only work with SMPL/SMPLH models for now.'))
        
        # Mean template vertices
        self.comp_register('init_v_template', torch.tensor(smpl_dict['v_template'][None], dtype=dtype), persistent=persistant_buffer)

        self.comp_register('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32), persistent=persistant_buffer)

        num_total_betas = smpl_dict['shapedirs'].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas
        
        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.comp_register('shapedirs', torch.tensor(shapedirs, dtype=dtype), persistent=persistant_buffer)

        if self.model_type == 'smplx':
            if smpl_dict['shapedirs'].shape[-1] > 300:
                begin_shape_id = 300
            else:
                begin_shape_id = 10
                num_expressions = smpl_dict['shapedirs'].shape[-1] - 10

            exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
            self.comp_register('exprdirs', torch.tensor(exprdirs, dtype=dtype), persistent=persistant_buffer)

            expression = torch.tensor(np.zeros((1, num_expressions)), dtype=dtype)
            self.comp_register('init_expression', expression, persistent=persistant_buffer)

        if self.use_dmpl:
            dmpldirs = np.load(dmpl_fname)['eigvec']

            dmpldirs = dmpldirs[:, :, :num_dmpls]
            self.comp_register('dmpldirs', torch.tensor(dmpldirs, dtype=dtype), persistent=persistant_buffer)

        # Regressor for joint locations given shape - 6890 x 24
        self.comp_register('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype), persistent=persistant_buffer)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.comp_register('posedirs', torch.tensor(posedirs, dtype=dtype), persistent=persistant_buffer)
        else:
            self.posedirs = None
        
        # indices of parents for each joints
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.comp_register('kintree_table', torch.tensor(kintree_table, dtype=torch.int32), persistent=persistant_buffer)

        # LBS weights
        # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        weights = smpl_dict['weights']
        self.comp_register('weights', torch.tensor(weights, dtype=dtype), persistent=persistant_buffer)

        self.comp_register('init_trans', torch.zeros((1,3), dtype=dtype), persistent=persistant_buffer)
        # self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        # root_orient
        # if self.model_type in ['smpl', 'smplh']:
        self.comp_register('init_root_orient', torch.zeros((1,3), dtype=dtype), persistent=persistant_buffer)

        # pose_body
        if self.model_type in ['smpl', 'smplh', 'smplx']:
            self.comp_register('init_pose_body', torch.zeros((1,63), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type == 'animal_horse':
            self.comp_register('init_pose_body', torch.zeros((1,105), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type == 'animal_dog':
            self.comp_register('init_pose_body', torch.zeros((1,102), dtype=dtype), persistent=persistant_buffer)

        # pose_hand
        if self.model_type in ['smpl']:
            self.comp_register('init_pose_hand', torch.zeros((1,1*3*2), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['smplh', 'smplx']:
            self.comp_register('init_pose_hand', torch.zeros((1,15*3*2), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['mano']:
            self.comp_register('init_pose_hand', torch.zeros((1,15*3), dtype=dtype), persistent=persistant_buffer)

        # face poses
        if self.model_type == 'smplx':
            self.comp_register('init_pose_jaw', torch.zeros((1,1*3), dtype=dtype), persistent=persistant_buffer)
            self.comp_register('init_pose_eye', torch.zeros((1,2*3), dtype=dtype), persistent=persistant_buffer)

        self.comp_register('init_betas', torch.zeros((1,num_betas), dtype=dtype), persistent=persistant_buffer)

        if self.use_dmpl:
            self.comp_register('init_dmpls', torch.zeros((1,num_dmpls), dtype=dtype), persistent=persistant_buffer)
    
    def r(self):
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        return c2c(self.forward().v)
    
    def comp_register(self, name, value, persistent=False):
        if sys.version_info[0] > 2:
            self.register_buffer(name, value, persistent)
        else:
            self.register_buffer(name, value)
    
    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, v_template=None, joints=None, v_shaped=None, return_dict=False,
                **kwargs):
        '''
        :param root_orient: Nx3
        :param pose_body:
        :param pose_hand:
        :param pose_jaw:
        :param pose_eye:
        :param kwargs:
        :return:
        '''
        batch_size = 1
        # compute batchsize by any of the provided variables
        for arg in [root_orient,pose_body,pose_hand,pose_jaw,pose_eye,betas,trans, dmpls,expression, v_template,joints]:
            if arg is not None:
                batch_size = arg.shape[0]
                break
        
        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'animal_horse', 'animal_dog'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano')
        if root_orient is None:  
            root_orient = self.init_root_orient.expand(batch_size, -1)

        if self.model_type in ['smplh', 'smpl']:
            if pose_body is None:  
                pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_hand is None:  
                pose_hand = self.init_pose_hand.expand(batch_size, -1)

        elif self.model_type == 'smplx':
            if pose_body is None:  
                pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_hand is None:  
                pose_hand = self.init_pose_hand.expand(batch_size, -1)
            if pose_jaw is None:  
                pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
            if pose_eye is None:  
                pose_eye = self.init_pose_eye.expand(batch_size, -1)

        elif self.model_type in ['mano',]:
            if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)

        elif self.model_type in ['animal_horse','animal_dog']:
            if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)

        if pose_hand is None and self.model_type not in ['animal_horse', 'animal_dog']:  
            pose_hand = self.init_pose_hand.expand(batch_size, -1)

        if trans is None: 
            trans = self.init_trans.expand(batch_size, -1)
        
        if v_template is None: 
            v_template = self.init_v_template.expand(batch_size, -1,-1)
        
        if betas is None: 
            betas = self.init_betas.expand(batch_size, -1)
        
        if self.model_type in ['smplh', 'smpl']:
            full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
        elif self.model_type == 'smplx':
            full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], dim=-1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
        elif self.model_type in ['mano', ]:
            full_pose = torch.cat([root_orient, pose_hand], dim=-1)
        elif self.model_type in ['animal_horse', 'animal_dog']:
            full_pose = torch.cat([root_orient, pose_body], dim=-1)
        
        if self.use_dmpl:
            if dmpls is None: dmpls = self.init_dmpls.expand(batch_size, -1)
            shape_components = torch.cat([betas, dmpls], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
        elif self.model_type == 'smplx':
            if expression is None: expression = self.init_expression.expand(batch_size, -1)
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs
        
        verts, Jtr = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                         shapedirs=shapedirs, posedirs=self.posedirs,
                         J_regressor=self.J_regressor, parents=self.kintree_table[0].long(),
                         lbs_weights=self.weights, joints=joints, v_shaped=v_shaped,
                         dtype=self.dtype)
        if kwargs["rm_ori_root"]:
            ori_jrt0 = Jtr[0, 0]
        else:
            ori_jrt0 = torch.zeros_like(Jtr[0, 0])
        
        trans = trans.unsqueeze(dim=1) - ori_jrt0
        Jtr = Jtr + trans
        verts = verts + trans

        res = {}
        res['v'] = verts
        res['f'] = self.f
        res['Jtr'] = Jtr 
        res['jtr_ori_root'] = ori_jrt0

        res['full_pose'] = full_pose
        res['trans'] = trans

        if not return_dict:
            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class
        
        return res






