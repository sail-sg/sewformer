import json
import numpy as np
import os
from pathlib import Path, PureWindowsPath
import shutil
import glob
from PIL import Image
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
import torchvision.transforms as T

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
root_path = os.path.dirname(os.path.dirname(os.path.abspath(parentdir)))
pkg_path = "{}/SewFactory/packages".format(root_path)
print(pkg_path)
sys.path.insert(0, pkg_path)

# My modules
from customconfig import Properties
from data.pattern_converter import NNSewingPattern, InvalidPatternDefError
import data.transforms as transforms
from data.panel_classes import PanelClasses

from data.human_body_prior.body_model import BodyModel
from data.utils import euler_angle_to_rot_6d


class GarmentDetrDataset(Dataset):
    def __init__(self, root_dir, sim_root, start_config, gt_caching=False, feature_caching=False, in_transforms=[]):

        self.root_path = root_dir
        self.sim_root = sim_root
        self.config = {}
        pattern_size_initialized = self.update_config(start_config)
        self.config['class'] = self.__class__.__name__
        self.datapoints_names = []
        self.dataset_start_ids = []  # (folder, start_id) tuples -- ordered by start id
        

        try:
            folders = [folder for folder in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, folder))]
            for folder in folders:
                self.dataset_start_ids.append((folder, len(self.datapoints_names)))
                if os.path.exists(os.path.join(self.root_path, folder, "renders")):
                    gt_folder = os.path.join(self.root_path, folder, "static")
                    img_names = [os.path.join(self.sim_root, folder, fn) for fn in os.listdir(os.path.join(self.root_path, folder, "renders")) if fn.endswith(".png")]
                    for img_name in img_names:
                        if os.path.exists(img_name):
                            merge_names = [(img_name, None, gt_folder)]
                            self.datapoints_names += merge_names
                else:
                    gt_folder = os.path.join(self.root_path, folder)
                    merge_names = [(None, None, gt_folder)]
                    self.datapoints_names += merge_names
        except Exception as e:
            print(e)
        
        self.dataset_start_ids.append((None, len(self.datapoints_names)))
        self.config['size'] = len(self)
        print("GarmentDetrDataset::Info::Total valid datanames is {}".format(self.config['size']))

        self.gt_cached, self.gt_caching = {}, gt_caching
        self.feature_cached, self.feature_caching = {}, feature_caching
        if self.gt_caching:
            print('GarmentDetrDataset::Info::Storing datapoints ground_truth info in memory')
        if self.feature_caching:
            print('GarmentDetrDataset::Info::Storing datapoints feature info in memory')
        
        self.color_transform = transforms.tv_make_color_img_transforms()
        self.geo_tranform = transforms.tv_make_geo_img_transforms(color=255)
        self.img_transform = transforms.tv_make_img_transforms()

        self.transforms = [transforms.SampleToTensor()] + in_transforms
        self.is_train = False
        self.gt_jsons = {"spec_dict":{}, "specs":{}}

        # Load panel classifier
        if self.config['panel_classification'] is not None:
            self.panel_classifier = PanelClasses(self.config['panel_classification'])
            self.config.update(max_pattern_len=len(self.panel_classifier))
        else:
            raise RuntimeError('GarmentDetrDataset::Error::panel_classification not found')
    
    def invalid_lists(self):
        invalid_fn = "./utilities/invalid_files.txt"
        with open(invalid_fn, "r") as f:
            invalid_lst = f.readlines()
        invalid_lst = [fn.strip() for fn in invalid_lst]
        invalid_lst.append("tee_SHPNN0VX1Z_wb_pants_straight_UBP5W37LKV")
        invalid_lst.append("jumpsuit_sleeveless_QI8XX8SQAQ")
        return invalid_lst
    
    def standardize(self, training=None):
        """Use shifting and scaling for fitting data to interval comfortable for NN training.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is already defined in config, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            configuration has a priority: if it's given, the statistics are NOT recalculated even if training set is provided
                => speed-up by providing stats or speeding up multiple calls to this function
        """
        print('GarmentDetrDataset::Using data normalization for features & ground truth')
        if 'standardize' in self.config:
            print('{}::Using stats from config'.format(self.__class__.__name__))
            stats = self.config['standardize']
        elif training is not None:
            # loader = DataLoader(training, batch_size=len(training), shuffle=False)
            training_indices = training.indices 
            gt_folders = self._load_gt_folders_from_indices(training_indices)
            outlines = []
            translations = []
            rotations = []
            stitch_tags = []
            for gt_folder in gt_folders:
                spec_dict = self._load_spec_dict(gt_folder)
                folder_elements = [os.path.basename(file) for file in glob.glob(os.path.join(gt_folder, "*"))]
                ground_truth = self._get_pattern_ground_truth(gt_folder, folder_elements, spec_dict)
                outlines.append(torch.Tensor(ground_truth['outlines']))
                translations.append(torch.Tensor(ground_truth['translations']))
                rotations.append(torch.Tensor(ground_truth['rotations']))
                stitch_tags.append(torch.Tensor(ground_truth['stitch_tags']))
        
            panel_shift, panel_scale = self._get_distribution_stats(torch.vstack(outlines), padded=True)
            # NOTE mean values for panels are zero due to loop property 
            # panel components SHOULD NOT be shifted to keep the loop property intact 
            panel_shift[0] = panel_shift[1] = 0
            # Use min\scale (normalization) instead of Gaussian stats for translation
            # No padding as zero translation is a valid value
            transl_min, transl_scale = self._get_norm_stats(torch.vstack(translations))
            rot_min, rot_scale = self._get_norm_stats(torch.vstack(rotations))
            # stitch tags if given
            st_tags_min, st_tags_scale = self._get_norm_stats(torch.vstack(stitch_tags))
        
            self.config['standardize'] = {
                'gt_shift': {
                    'outlines': panel_shift.cpu().numpy(), 
                    'aug_outlines': panel_shift.cpu().numpy(),
                    'rotations': rot_min.cpu().numpy(),
                    'translations': transl_min.cpu().numpy(), 
                    'stitch_tags': st_tags_min.cpu().numpy()
                },
                'gt_scale': {
                    'outlines': panel_scale.cpu().numpy(), 
                    "aug_outlines": panel_scale.cpu().numpy(),
                    'rotations': rot_scale.cpu().numpy(),
                    'translations': transl_scale.cpu().numpy(),
                    'stitch_tags': st_tags_scale.cpu().numpy()
                }
            }
            stats = self.config['standardize']
        else:
            raise ValueError('GarmentDetrDataset::Error::Standardization cannot be applied: supply either stats in config or training set to use standardization')
        print(stats)
        
        # clean-up tranform list to avoid duplicates
        self.transforms = [t for t in self.transforms if not isinstance(t, transforms.GTtandartization) and not isinstance(t, transforms.FeatureStandartization)]

        self.transforms.append(transforms.GTtandartization(stats['gt_shift'], stats['gt_scale']))

    def save_to_wandb(self, experiment):
        """Save data cofiguration to current expetiment run"""
        # config
        experiment.add_config('dataset', self.config)
        # panel classes
        if self.panel_classifier is not None:
            shutil.copy(
                self.panel_classifier.filename, 
                experiment.local_wandb_path() / ('panel_classes.json'))
    
    def set_training(self, is_train=True):
        self.is_train = is_train
    
    def update_config(self, in_config):
        """Define dataset configuration:
            * to be part of experimental setup on wandb
            * Control obtainign values for datapoints"""

        # initialize keys for correct dataset initialization
        if ('max_pattern_len' not in in_config
                or 'max_panel_len' not in in_config
                or 'max_num_stitches' not in in_config):
            in_config.update(max_pattern_len=None, max_panel_len=None, max_num_stitches=None)
            pattern_size_initialized = False
        else:
            pattern_size_initialized = True
        
        if 'obj_filetag' not in in_config:
            in_config['obj_filetag'] = ''  # look for objects with this tag in filename when loading 3D models

        if 'panel_classification' not in in_config:
            in_config['panel_classification'] = None
        
        self.config.update(in_config)
        # check the correctness of provided list of datasets
        if ('data_folders' not in self.config 
                or not isinstance(self.config['data_folders'], list)
                or len(self.config['data_folders']) == 0):
            #raise RuntimeError('BaseDataset::Error::information on datasets (folders) to use is missing in the incoming config')
            print(f'{self.__class__.__name__}::Info::Collecting all datasets (no sub-folders) to use')
        return pattern_size_initialized

    def __len__(self, ):
        """Number of entries in the dataset"""
        return len(self.datapoints_names) 

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        rst = T.Resize(512)
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        folder_elements = None  
        datapoint_name, smpl_name, gt_folder = self.datapoints_names[idx]

        img, ground_truth, smpl_uv = self._get_sample_info(datapoint_name, gt_folder, smpl_name)
        img = rst(img)
        name = os.path.basename(os.path.dirname(gt_folder))
        folder = os.path.dirname(gt_folder)

        if "use_smpl_loss" in self.config and  self.config["use_smpl_loss"]:
            smpl_pos_fn = self.get_smpl_pose_fn(datapoint_name, gt_folder)
            smpl_joint_pose = json.load(open(smpl_pos_fn, "r"))["pose"]
            smpl_joint_pose = [euler_angle_to_rot_6d(p) for p in smpl_joint_pose[1:23]]
            smpl_joint_pose = np.array(smpl_joint_pose).squeeze()
            ground_truth.update({"smpl_joints": smpl_joint_pose})
        
        if self.is_train and self.config["augment"]:
            img_tensor = self.img_transform(self.geo_tranform(img))
        else:
            img_tensor = self.img_transform(img)
        
        # stitches
        if ground_truth["stitch_adj"] is not None:
            masked_stitches, stitch_edge_mask, reindex_stitches = self.match_edges(ground_truth["free_edges_mask"], \
                                                                                stitches=ground_truth["stitches"], \
                                                                                max_num_stitch_edges=self.config["max_stitch_edges"])
            label_indices = self.split_pos_neg_pairs(reindex_stitches, num_max_edges=1000)

            ground_truth.update({"masked_stitches": masked_stitches,
                                 "stitch_edge_mask": stitch_edge_mask,
                                 "reindex_stitches": reindex_stitches,
                                 "label_indices": label_indices})
        
        sample = {'image': img_tensor, 
                  'ground_truth': ground_truth, 
                  'name': name, 
                  'data_folder': folder, 
                  'img_fn': datapoint_name}
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
    def _swap_name(self, name):
        # sim root to original root
        name = name.replace(self.sim_root, self.root_path).split('/')
        return '/'.join(name[:-1] + ["renders"] + name[-1:])
    
    def _load_gt_folders_from_indices(self, indices):
        gt_folders = [self.datapoints_names[idx][-1] for idx in indices]
        return list(set(gt_folders))
    
    def get_smpl_pose_fn(self, datapoint_name, gt_folder):
        return os.path.join(os.path.dirname(gt_folder), "poses", os.path.basename(datapoint_name).split("_")[0] + "__body_info.json")

    
    def _drop_cache(self):
        """Clean caches of datapoints info"""
        self.gt_cached = {}
        self.feature_cached = {}
    
    def _renew_cache(self):
        """Flush the cache and re-fill it with updated information if any kind of caching is enabled"""
        self.gt_cached = {}
        self.feature_cached = {}
        if self.feature_caching or self.gt_caching:
            for i in range(len(self)):
                self[i]
            print('Data cached!')
    
    def random_split_by_dataset(self, valid_per_type, test_per_type, split_type='count', split_on="img"):
        if split_type != 'count' and split_type != 'percent':
            raise NotImplementedError('{}::Error::Unsupported split type <{}> requested'.format(
                self.__class__.__name__, split_type))
        train_ids, valid_ids, test_ids = [], [], []

        if split_on == "img":
            data_len = len(self)
            permute = (torch.randperm(data_len)).tolist()
            valid_size = int(data_len * valid_per_type / 100) if split_type == 'percent' else valid_per_type
            test_size = int(data_len * test_per_type / 100) if split_type == 'percent' else test_per_type
            train_size = data_len - valid_size - test_size
            train_ids, valid_ids = permute[:train_size], permute[train_size:train_size + valid_size]
            if test_size:
                test_ids = permute[train_size + valid_size:train_size + valid_size + test_size]
        elif split_on == "folder":
            # valid_per_type & test_per_type used on folder
            data_len = len(self.dataset_start_ids) - 1
            permute = (torch.randperm(data_len)).tolist()
            valid_size = int(data_len * valid_per_type / 100) if split_type == 'percent' else valid_per_type
            test_size = int(data_len * test_per_type / 100) if split_type == 'percent' else test_per_type
            train_size = data_len - valid_size - test_size
            train_folders, valid_folders = permute[:train_size], permute[train_size:train_size + valid_size]
            if test_size:
                test_folders = permute[train_size + valid_size:train_size + valid_size + test_size]
            
            for idx in range(len(self.dataset_start_ids) - 1):
                start_id = self.dataset_start_ids[idx][1] 
                end_id = self.dataset_start_ids[idx + 1][1]
                data_len = end_id - start_id
                if idx in train_folders:
                    train_ids.extend((torch.randperm(data_len) + start_id).tolist())
                elif idx in valid_folders:
                    valid_ids.extend((torch.randperm(data_len) + start_id).tolist())
                elif test_size and idx in test_folders:
                    test_ids.extend((torch.randperm(data_len) + start_id).tolist())
                else:
                    raise ValueError('GarmentDetrDataset::Error::Run on data split with split_on==folder')
            
            random.shuffle(train_ids)
            random.shuffle(valid_ids)
            if test_size:
                random.shuffle(test_ids)
        return (
            Subset(self, train_ids), 
            Subset(self, valid_ids),
            Subset(self, test_ids) if test_size else None
        )

    def get_simple_names(self, dataname, data_root=""):
        return os.path.dirname(dataname[-1].replace(data_root, ""))
    
    def split_from_dict(self, split_dict):
        """
            Reproduce the data split in the provided dictionary: 
            the elements of the currect dataset should play the same role as in provided dict
        """
        split_dict_root = ""
        train_ids, valid_ids, test_ids = [], [], []
        
        training_datanames = split_dict['train'] if "train" in split_dict else split_dict["training"]
        training_datanames = [self.get_simple_names(dataname, data_root=split_dict_root) for dataname in training_datanames]
        valid_datanames = split_dict['validation']
        valid_datanames = [self.get_simple_names(dataname, data_root=split_dict_root) for dataname in valid_datanames]
        test_datanames = split_dict['test']
        test_datanames = [self.get_simple_names(dataname, data_root=split_dict_root) for dataname in test_datanames]

        dataset_root = ""
        if dataset_root in training_datanames[0]:
            dataset_root = ""
        for idx in range(len(self.datapoints_names)):
            if self.get_simple_names(self.datapoints_names[idx], data_root=dataset_root) in training_datanames:  # usually the largest, so check first
                train_ids.append(idx)
            elif len(test_datanames) > 0 and self.get_simple_names(self.datapoints_names[idx], data_root=dataset_root) in test_datanames:
                test_ids.append(idx)
            elif len(valid_datanames) > 0 and self.get_simple_names(self.datapoints_names[idx], data_root=dataset_root) in valid_datanames:
                valid_ids.append(idx)
            else:
                continue
            
            if idx % 1000 == 0:
                print(f"progress {idx}, #Train_Ids={len(train_ids)}, #Valid_Ids={len(valid_ids)}, #Test_Ids={len(test_ids)}")
        
        return (
            Subset(self, train_ids), 
            Subset(self, valid_ids),
            Subset(self, test_ids) if len(test_ids) > 0 else None
        )


    # ----- Sample -----
    def _get_sample_info(self, datapoint_name, gt_folder, smpl_name=None):
        """
            Get features and Ground truth prediction for requested data example
        """
        folder_elements = [os.path.basename(file) for file in glob.glob(os.path.join(gt_folder, "*"))]  # all files in this directory
        if datapoint_name in self.feature_cached:
            image = self.feature_cached[datapoint_name]
        else:
            try:
                image = Image.open(datapoint_name).convert('RGB')
            except Exception as e:
                image = Image.open(self._swap_name(datapoint_name)).convert('RGB')
            if self.feature_caching:
                self.feature_cached[datapoint_name] = image

        # GT -- pattern
        if gt_folder in self.gt_cached: # might not be compatible with list indexing
            ground_truth = self.gt_cached[gt_folder]
        else:
            spec_dict = self._load_spec_dict(gt_folder)
            ground_truth = self._get_pattern_ground_truth(gt_folder, folder_elements, spec_dict)
            if self.gt_caching:
                self.gt_cached[gt_folder] = ground_truth
        return image, ground_truth, None
    
    def _load_spec_dict(self, gt_folder):
        if gt_folder in self.gt_jsons["spec_dict"]:
            return self.gt_jsons["spec_dict"][gt_folder]
        else:
            # add smpl root at static pose
            static_pose = json.load(open(gt_folder + "/static__body_info.json", "r"))
            static_root = static_pose["trans"]
            spec_dict = json.load(open(gt_folder + "/spec_config.json", "r"))
            for key, val in spec_dict.items():
                spec = PureWindowsPath(val["spec"]).parts[-1]
                spec_dict[key]["spec"] = spec
                spec_dict[key]["delta"] = np.array(val["delta"]) - np.array(static_root)
            self.gt_jsons["spec_dict"][gt_folder] = spec_dict
            return spec_dict
    
    def _get_pattern_ground_truth(self, gt_folder, folder_elements, spec_dict):
        """Get the pattern representation with 3D placement"""
        patterns = self._read_pattern(
            gt_folder, folder_elements, spec_dict,
            pad_panels_to_len=self.config['max_panel_len'],
            pad_panel_num=self.config['max_pattern_len'],
            pad_stitches_num=self.config['max_num_stitches'],
            with_placement=True, with_stitches=True, with_stitch_tags=True)
        pattern, num_edges, num_panels, rots, tranls, stitches, num_stitches, stitch_adj, stitch_tags, aug_outlines = patterns 
        free_edges_mask = self.free_edges_mask(pattern, stitches, num_stitches)
        empty_panels_mask = self._empty_panels_mask(num_edges)  # useful for evaluation
        
        ground_truth = {
            'outlines': pattern, 'num_edges': num_edges,
            'rotations': rots, 'translations': tranls, 
            'num_panels': num_panels, 'empty_panels_mask': empty_panels_mask, 'num_stitches': num_stitches,
            'stitches': stitches, 'stitch_adj': stitch_adj, 'free_edges_mask': free_edges_mask, 'stitch_tags': stitch_tags
        }

        if aug_outlines[0] is not None:
            ground_truth.update({"aug_outlines": aug_outlines})

        return ground_truth
    
    def _load_ground_truth(self, gt_folder):
        folder_elements = [os.path.basename(file) for file in glob.glob(os.path.join(gt_folder, "*"))] 
        spec_dict = self._load_spec_dict(gt_folder)
        ground_truth = self._get_pattern_ground_truth(gt_folder, folder_elements, spec_dict)
        return ground_truth
    
    def _empty_panels_mask(self, num_edges):
        """Empty panels as boolean mask"""

        mask = np.zeros(len(num_edges), dtype=bool)
        mask[num_edges == 0] = True

        return mask
    
    @staticmethod
    def match_edges(free_edge_mask, stitches=None, max_num_stitch_edges=56):
        stitch_edges = np.ones((1, max_num_stitch_edges)) * (-1)
        valid_edges = (~free_edge_mask.reshape(-1)).nonzero()
        stitch_edge_mask = np.zeros((1, max_num_stitch_edges))
        if stitches is not None:
            stitches = np.transpose(stitches)
            reindex_stitches = np.zeros((1, max_num_stitch_edges, max_num_stitch_edges))
        else:
            reindex_stitches = None
        
        batch_edges = valid_edges[0]
        num_edges = batch_edges.shape[0]
        stitch_edges[:, :num_edges] = batch_edges
        stitch_edge_mask[:, :num_edges] = 1
        if stitches is not None:
            for stitch in stitches:
                side_i, side_j = stitch
                if side_i != -1 and side_j != -1:
                    reindex_i, reindex_j = np.where(stitch_edges[0] == side_i)[0], np.where(stitch_edges[0] == side_j)[0]
                    reindex_stitches[0, reindex_i, reindex_j] = 1
                    reindex_stitches[0, reindex_j, reindex_i] = 1
        
        return stitch_edges * stitch_edge_mask, stitch_edge_mask, reindex_stitches
    
    @staticmethod
    def split_pos_neg_pairs(stitches, num_max_edges=3000):
        stitch_ind = np.triu_indices_from(stitches[0], 1)
        pos_ind = [[stitch_ind[0][i], stitch_ind[1][i]] for i in range(stitch_ind[0].shape[0]) if stitches[0, stitch_ind[0][i], stitch_ind[1][i]]]
        neg_ind = [[stitch_ind[0][i], stitch_ind[1][i]] for i in range(stitch_ind[0].shape[0]) if not stitches[0, stitch_ind[0][i], stitch_ind[1][i]]]

        assert len(neg_ind) >= num_max_edges
        neg_ind = neg_ind[:num_max_edges]
        pos_inds = np.expand_dims(np.array(pos_ind), axis=1)
        neg_inds = np.repeat(np.expand_dims(np.array(neg_ind), axis=0), repeats=pos_inds.shape[0], axis=0)
        indices = np.concatenate((pos_inds, neg_inds), axis=1)
        return indices
    

    # ------------- Datapoints Utils --------------
    def template_name(self, spec):
        """Get name of the garment template from the path to the datapoint"""
        return "_".join(spec.split('_')[:-1]) 
    
    def _read_pattern(self, gt_folder, folder_elements, spec_dict,
                      pad_panels_to_len=None, pad_panel_num=None, pad_stitches_num=None,
                      with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Read given pattern in tensor representation from file"""
        

        spec_list = {}
        for key, val in spec_dict.items():
            spec_file = [file for file in folder_elements if val["spec"] in file and "specification.json" in file]
            if len(spec_file) > 0:
                spec_list[key] = spec_file[0]
            else:
                raise ValueError("Specification Cannot be found in folder_elements for {}".format(gt_folder))
        
        if not spec_list:
            raise RuntimeError('GarmentDetrDataset::Error::*specification.json not found for {}'.format(gt_folder))
        patterns = []

        for key, spec in spec_list.items():
            if gt_folder + "/" + spec in self.gt_jsons["specs"]:
                pattern = self.gt_jsons["specs"][gt_folder + "/" + spec]
            else:
                pattern = NNSewingPattern(
                    gt_folder + "/" + spec, 
                    panel_classifier=self.panel_classifier, 
                    template_name=self.template_name(spec_dict[key]['spec']))
                self.gt_jsons["specs"][gt_folder + "/" + spec] = pattern
            patterns.append(pattern)

        pat_tensor = NNSewingPattern.multi_pattern_as_tensors(patterns,
            pad_panels_to_len, pad_panels_num=pad_panel_num, pad_stitches_num=pad_stitches_num,
            with_placement=with_placement, with_stitches=with_stitches, 
            with_stitch_tags=with_stitch_tags, spec_dict=spec_dict)
        return pat_tensor
    
    
    def get_item_infos(self, idx):
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()
        datapoint_name, smpl_name, gt_folder = self.datapoints_names[idx]
        data_prop_fn = os.path.join(os.path.dirname(gt_folder), "data_props.json")
        with open(data_prop_fn, 'r') as f:
            data_props = json.load(f)
        pose_fbx = data_props["body"]["name"].replace(data_props["body_path"] + "\\", "")
        spec_fns = list(data_props["garments"]["config"]["pattern_specs"].values())
        return pose_fbx, spec_fns, (datapoint_name, gt_folder)
    
    def _unpad(self, element, tolerance=1.e-5):
        """Return copy of input element without padding from given element. Used to unpad edge sequences in pattern-oriented datasets"""
        # NOTE: might be some false removal of zero edges in the middle of the list.
        if torch.is_tensor(element):        
            bool_matrix = torch.isclose(element, torch.zeros_like(element), atol=tolerance)  # per-element comparison with zero
            selection = ~torch.all(bool_matrix, axis=1)  # only non-zero rows
        else:  # numpy
            selection = ~np.all(np.isclose(element, 0, atol=tolerance), axis=1)  # only non-zero rows
        return element[selection]
    
    def _get_distribution_stats(self, input_batch, padded=False):
        """Calculates mean & std values for the input tenzor along the last dimention"""

        input_batch = input_batch.view(-1, input_batch.shape[-1])
        if padded:
            input_batch = self._unpad(input_batch)  # remove rows with zeros

        # per dimention means
        mean = input_batch.mean(axis=0)
        # per dimention stds
        stds = ((input_batch - mean) ** 2).sum(0)
        stds = torch.sqrt(stds / input_batch.shape[0])

        return mean, stds
    
    def _get_norm_stats(self, input_batch, padded=False):
        """Calculate shift & scaling values needed to normalize input tenzor 
            along the last dimention to [0, 1] range"""
        input_batch = input_batch.view(-1, input_batch.shape[-1])
        if padded:
            input_batch = self._unpad(input_batch)  # remove rows with zeros

        # per dimention info
        min_vector, _ = torch.min(input_batch, dim=0)
        max_vector, _ = torch.max(input_batch, dim=0)
        scale = torch.empty_like(min_vector)

        # avoid division by zero
        for idx, (tmin, tmax) in enumerate(zip(min_vector, max_vector)): 
            if torch.isclose(tmin, tmax):
                scale[idx] = tmin if not torch.isclose(tmin, torch.zeros(1)) else 1.
            else:
                scale[idx] = tmax - tmin
        
        return min_vector, scale
    
    # ----- Saving predictions -----
    @staticmethod
    def free_edges_mask(pattern, stitches, num_stitches):
        """
        Construct the mask to identify edges that are not connected to any other
        """
        mask = np.ones((pattern.shape[0], pattern.shape[1]), dtype=bool)
        max_edge = pattern.shape[1]

        for side in stitches[:, :num_stitches]:  # ignore the padded part
            for edge_id in side:
                mask[edge_id // max_edge][edge_id % max_edge] = False
        
        return mask
    
    @staticmethod
    def prediction_to_stitches(free_mask_logits, similarity_matrix, return_stitches=False):
        free_mask = (torch.sigmoid(free_mask_logits.squeeze(-1)) > 0.5).flatten()
        if not return_stitches:
            simi_matrix = similarity_matrix + similarity_matrix.transpose(0, 1)
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(0), -float("inf"))
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(-1), 0)
            num_stitches = free_mask.nonzero().shape[0] // 2
        else:
            simi_matrix = similarity_matrix
            num_stitches = simi_matrix.shape[0] // 2
        simi_matrix = torch.triu(simi_matrix, diagonal=1)
        stitches = []
        
        for i in range(num_stitches):
            index = (simi_matrix == torch.max(simi_matrix)).nonzero()
            stitches.append((index[0, 0].cpu().item(), index[0, 1].cpu().item()))
            simi_matrix[index[0, 0], :] = -float("inf")
            simi_matrix[index[0, 1], :] = -float("inf")
            simi_matrix[:, index[0, 0]] = -float("inf")
            simi_matrix[:, index[0, 1]] = -float("inf")
        
        if len(stitches) == 0:
            stitches = None
        else:
            stitches = np.array(stitches)
            if stitches.shape[0] != 2:
                stitches = np.transpose(stitches, (1, 0))
        return stitches


    def save_gt_batch_imgs(self, gt_batch, datanames, data_folders, save_to):
        gt_imgs = []
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):
            gt = {}
            for key in gt_batch:
                gt[key] = gt_batch[key][idx]
                if (('order_matching' in self.config and self.config['order_matching'])
                    or 'origin_matching' in self.config and self.config['origin_matching']
                    or not self.gt_caching):
                    print(f'{self.__class__.__name__}::Warning::Propagating '
                        'information from GT on prediction is not implemented in given context')
                else:
                    if self.gt_caching and folder + '/static' in self.gt_cached:
                        gtc = self.gt_cached[folder + '/static']
                    else:
                        gtc = self._load_ground_truth(folder + "/static")
                    for key in gtc:
                        if key not in gt:
                            gt[key] = gtc[key]
            
            # Transform to pattern object
            pname = os.path.basename(folder) + "__" + os.path.basename(name.replace(".png", ""))
            pattern = self._pred_to_pattern(gt, pname)

            try: 
                # log gt number of panels
                # pattern.spec['properties']['correct_num_panels'] = gtc['num_panels']
                final_dir = pattern.serialize(save_to, to_subfolder=True, tag=f'_gt_')
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(name, e))
                continue

            final_file = pattern.name + '_gt__pattern.png'
            gt_imgs.append(Path(final_dir) / final_file)
        return gt_imgs
    
    def save_prediction_single(self, prediction, dataname="outside_dataset", save_to=None, return_stitches=False):
        

        for key in prediction.keys():
            prediction[key] = prediction[key][0]
        
        pattern = self._pred_to_pattern(prediction, dataname, return_stitches=return_stitches)
        try: 
            final_dir = pattern.serialize(save_to, to_subfolder=True, tag='_predicted_single_')
        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(dataname, e))

        final_file = pattern.name + '_predicted__pattern.png'
        prediction_img = Path(final_dir) / final_file
        
        return pattern.pattern['panel_order'], pattern.pattern['new_panel_ids'], prediction_img


    def save_prediction_batch(self, predictions, datanames, data_folders, 
            save_to, features=None, weights=None, orig_folder_names=False, **kwargs):
        """ 
            Saving predictions on batched from the current dataset
            Saves predicted params of the datapoint to the requested data folder.
            Returns list of paths to files with prediction visualizations
            Assumes that the number of predictions matches the number of provided data names"""

        prediction_imgs = []
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):
            # "unbatch" dictionary
            prediction = {}
            pname = os.path.basename(folder) + "__" + os.path.basename(name.replace(".png", ""))
            tmp_path = os.path.join(save_to, pname, '_predicted_specification.json')
            if os.path.exists(tmp_path):
                continue
            
            print("Progress {}".format(tmp_path))

            for key in predictions:
                prediction[key] = predictions[key][idx]
            if "images" in kwargs:
                prediction["input"] = kwargs["images"][idx]
            if "panel_shape" in kwargs:
                prediction["panel_l2"] = kwargs["panel_shape"][idx]

            # add values from GT if not present in prediction
            if "use_gt_stitches" in kwargs and kwargs["use_gt_stitches"]:
                gt = self._load_ground_truth(folder + "/static")
                for key in gt:
                    if key not in prediction:
                        prediction[key] = gt[key]
            else:
                if "edge_cls" in prediction and "edge_similarity" in prediction:
                    print("Use the predicted stitch infos ")
                else:
                    gt = self.gt_cached[folder + '/static']
                    for key in gt:
                        if key not in prediction:
                            prediction[key] = gt[key]
            # Transform to pattern object
            
            pattern = self._pred_to_pattern(prediction, pname)
            # log gt number of panels
            if self.gt_caching and folder + "/static" in self.gt_cached:
                gt = self.gt_cached[folder + '/static']
                pattern.spec['properties']['correct_num_panels'] = gt['num_panels']
            elif "use_gt_stitches" in kwargs and kwargs["use_gt_stitches"]:
                pattern.spec['properties']['correct_num_panels'] = gt['num_panels']

            try: 
                tag = f'_predicted_{prediction["panel_l2"]}_' if "panel_l2" in prediction else f"_predicted_"
                final_dir = pattern.serialize(save_to, to_subfolder=True, tag=tag)
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(folder, e))
                continue
            final_file = pattern.name + '_predicted__pattern.png'
            prediction_imgs.append(Path(final_dir) / final_file)
            # save input img
            T.ToPILImage()(prediction["input"]).save(os.path.join(final_dir, "input.png")) 
            shutil.copy2(name, str(final_dir))
            # shutil.copy2(name.replace(".png", "cam_pos.json"), str(final_dir))
            shutil.copy2(os.path.join(folder, "static", "spec_config.json"), str(final_dir))

            # copy originals for comparison
            data_prop_file = os.path.join(folder, "data_props.json")
            if os.path.exists(data_prop_file):
                shutil.copy2(data_prop_file, str(final_dir))
        return prediction_imgs
    
    def _pred_to_pattern(self, prediction, dataname, return_stitches=False):
        """Convert given predicted value to pattern object
        """
        # undo standardization  (outside of generinc conversion function due to custom std structure)
        gt_shifts = self.config['standardize']['gt_shift']
        gt_scales = self.config['standardize']['gt_scale']

        for key in gt_shifts:
            if key == 'stitch_tags':  
                # ignore stitch tags update if explicit tags were not used
                continue
            
            pred_numpy = prediction[key].detach().cpu().numpy()
            if key == 'outlines' and len(pred_numpy.shape) == 2: 
                pred_numpy = pred_numpy.reshape(self.config["max_pattern_len"], self.config["max_panel_len"], 4)

            prediction[key] = pred_numpy * gt_scales[key] + gt_shifts[key]

        # recover stitches
        if 'stitches' in prediction:  # if somehow prediction already has an answer
            stitches = prediction['stitches']
        elif 'stitch_tags' in prediction: # stitch tags to stitch list 
            pass
        elif 'edge_cls' in prediction and "edge_similarity" in prediction:
            stitches = self.prediction_to_stitches(prediction['edge_cls'], prediction['edge_similarity'], return_stitches=return_stitches)
        else:
            stitches = None
        
        # Construct the pattern from the data
        pattern = NNSewingPattern(view_ids=False, panel_classifier=self.panel_classifier)
        pattern.name = dataname

        try: 
            pattern.pattern_from_tensors(
                prediction['outlines'], 
                panel_rotations=prediction['rotations'],
                panel_translations=prediction['translations'], 
                stitches=stitches,
                padded=True)   
        except (RuntimeError, InvalidPatternDefError) as e:
            print('GarmentDetrDataset::Warning::{}: {}'.format(dataname, e))
            pass
        
        return pattern
    

if __name__ == '__main__':
    from data.wrapper import RealisticDatasetDetrWrapper
    import pdb; pdb.set_trace()
    system = Properties('./system.json')
    start_config = {"panel_classification": "./data_configs/panel_classes_condenced.json",
                    "max_pattern_len": 23, "max_panel_len": 14, "max_num_stitches": 28,
                    "max_stitch_edges": 56, "element_size": 4, "rotation_size": 4, "translation_size": 3,
                    "use_sim": True, "use_smpl_loss": True, "augment": True}
    dataset = GarmentDetrDataset(system['datasets_path'], None, start_config)
    example_data = dataset[0]
    split_info = {"type": "percent", "split_on": "folder", "valid_per_type": 5, "test_per_type": 10}
    datawrapper = RealisticDatasetDetrWrapper(dataset, batch_size=64)
    datawrapper.load_split(split_info, batch_size=64)
    datawrapper.standardize_data()




    



