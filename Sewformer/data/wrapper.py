from argparse import Namespace
import json
import numpy as np
import random
import time
from datetime import datetime
import os

import torch
from torch.utils.data import DataLoader, Subset

# My modules
import data.transforms as transforms

def collate_fn(batches):
    # start_time = time.time()
    if isinstance(batches[0]["ground_truth"], dict):
        bdict = {key: [] for key in batches[0].keys()}
        bdict["ground_truth"] = {key:[] for key in batches[0]["ground_truth"]}
        cum_sum = 56
        for i, batch in enumerate(batches):
            for key, val in batch.items():
                if key in ["image", "name", "data_folder", "img_fn"]:
                    bdict[key].append(val)
                else:
                    for k, v in batch["ground_truth"].items():
                        if k != "label_indices":
                            bdict["ground_truth"][k].append(v)
                        else:
                            new_label_indices = v.clone()
                            new_label_indices[:, :, 0] += cum_sum * i
                            bdict["ground_truth"][k].append(new_label_indices)
        
        for key in bdict.keys():
            if key == "image":
                bdict[key] = torch.stack(bdict[key])
            elif key == "ground_truth":
                for k in bdict[key]:
                    if k in ["label_indices", "masked_stitches", "stitch_edge_mask", "reindex_stitches"]:
                        bdict[key][k] = torch.vstack(bdict[key][k])
                    else:
                        bdict[key][k] = torch.stack(bdict[key][k])
        # print("collate_fn: {}".format(time.time() - start_time))
        return bdict
    else:
        bdict = {key: [] for key in batches[0].keys()}
        for i, batch in enumerate(batches):
            for key, val in batch.items():
                bdict[key].append(val)
        bdict["features"] = torch.stack(bdict["features"])
        bdict["ground_truth"] = torch.stack(bdict["ground_truth"])
        return bdict


class RealisticDatasetDetrWrapper(object):
    """Resposible for keeping dataset, its splits, loaders & processing routines.
        Allows to reproduce earlier splits
    """

    def __init__(self, in_dataset, known_split=None, batch_size=None, shuffle_train=True):
        
        self.dataset = in_dataset
        self.data_section_list = ['full', 'train', 'validation', 'test']

        self.training = in_dataset
        self.validation = None
        self.test = None

        self.batch_size = None

        self.loaders = Namespace(
            full=None,
            train=None,
            test=None,
            real_test=None,
            validation=None
        )

        self.split_info = {
            'random_seed': None, 
            'valid_per_type': None, 
            'test_per_type': None,
            'split_on': None
        }

        if known_split is not None:
            self.load_split(known_split)
        if batch_size is not None:
            self.batch_size = batch_size
            self.new_loaders(batch_size, shuffle_train)
        self.get_real_test_ids(batch_size)
    
    def get_loader(self, data_section='full'):
        """Return loader that corresponds to given data section. None if requested loader does not exist"""
        try:
            return getattr(self.loaders, data_section)
        except AttributeError:
            raise ValueError('RealisticDataWrapper::requested loader on unknown data section {}'.format(data_section))
    
    def new_loaders(self, batch_size=None, shuffle_train=True, multiprocess=False):
        """Create loaders for current data split. Note that result depends on the random number generator!
        
            if the data split was not specified, only the 'full' loaders are created
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is None:
            raise RuntimeError('RealisticDatasetDetrWrapper:Error:cannot create loaders: batch_size is not set')
        
        if multiprocess:
            full_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            self.loaders.full = DataLoader(self.dataset, self.batch_size, shuffle=False, num_workers=0, pin_memory=True,
                                        sampler=full_sampler)
        else:
            self.loaders.full = DataLoader(self.dataset, self.batch_size)
        if self.validation is not None and self.test is not None:
        # if True:
            # we have a loaded split!
            
            print('{}::Warning::Failed to create balanced batches for training. Using default sampling'.format(self.__class__.__name__))
            self.dataset.config['balanced_batch_sampling'] = False

            if multiprocess:
                train_sampler = torch.utils.data.distributed.DistributedSampler(self.training, drop_last=True)
                self.loaders.train = DataLoader(self.training, self.batch_size, 
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=12,
                                                collate_fn=collate_fn,
                                                sampler=train_sampler)
            else:
                print("No Multiprocess")
                self.loaders.train = DataLoader(self.training, self.batch_size, 
                                                #collate_fn=collate_fn, 
                                                num_workers=12, 
                                                pin_memory=True,
                                                shuffle=shuffle_train)

            self.loaders.validation = DataLoader(self.validation, self.batch_size, collate_fn=collate_fn, num_workers=12)
            self.loaders.test = DataLoader(self.test, self.batch_size, collate_fn=collate_fn, num_workers=12)
        return self.loaders.train, self.loaders.validation, self.loaders.test

    def _loaders_dict(self, subsets_dict, batch_size, shuffle=False):
        """Create loaders for all subsets in dict"""
        loaders_dict = {}
        for name, subset in subsets_dict.items():
            loaders_dict[name] = DataLoader(subset, batch_size, shuffle=shuffle)
        return loaders_dict
    
    # -------- Reproducibility ---------------
    def new_split(self, valid, test=None, random_seed=None):
        """Creates train/validation or train/validation/test splits
            depending on provided parameters
            """
        self.split_info['random_seed'] = random_seed if random_seed else int(time.time())
        self.split_info.update(valid_per_type=valid, test_per_type=test, type='count', split_on='folder')
        
        return self.load_split()
    
    
    def load_split(self, split_info=None, batch_size=None):
        """Get the split by provided parameters. Can be used to reproduce splits on the same dataset.
            NOTE this function re-initializes torch random number generator!
        """

        if split_info:
            self.split_info = split_info
        
        if 'random_seed' not in self.split_info or self.split_info['random_seed'] is None:
            self.split_info['random_seed'] = int(time.time())
        # init for all libs =)
        torch.manual_seed(self.split_info['random_seed'])
        random.seed(self.split_info['random_seed'])
        np.random.seed(self.split_info['random_seed'])

        # if file is provided
        if 'filename' in self.split_info and self.split_info['filename'] is not None:
            print('RealisticDatasetDetrWrapper::Loading data split from {}'.format(self.split_info['filename']))
            with open(self.split_info['filename'], 'r') as f_json:
                split_dict = json.load(f_json)
            self.training, self.validation, self.test = self.dataset.split_from_dict(
                split_dict)
        else:
            keys_required = ['test_per_type', 'valid_per_type', 'type', 'split_on']
            if any([key not in self.split_info for key in keys_required]):
                raise ValueError('Specified split information is not full: {}. It needs to contain: {}'.format(split_info, keys_required))
            print('RealisticDatasetDetrWrapper::Loading data split from split config: {}: valid per type {} / test per type {}'.format(
                self.split_info['type'], self.split_info['valid_per_type'], self.split_info['test_per_type']))
            self.training, self.validation, self.test = self.dataset.random_split_by_dataset(
                self.split_info['valid_per_type'], 
                self.split_info['test_per_type'],
                self.split_info['type'], 
                self.split_info['split_on'])

        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is not None:

            self.new_loaders()  # s.t. loaders could be used right away
        
        print('RealisticDatasetDetrWrapper::Dataset split: {} / {} / {}'.format(
            len(self.training) if self.training else None, 
            len(self.validation) if self.validation else None, 
            len(self.test) if self.test else None))
        
        self.split_info['size_train'] = len(self.training) if self.training else 0
        self.split_info['size_valid'] = len(self.validation) if self.validation else 0
        self.split_info['size_test'] = len(self.test) if self.test else 0

        self.get_data_lists(self.training, self.validation, self.test, self.split_info)
        
        return self.training, self.validation, self.test
    
    def get_data_lists(self, training, validation, testing, split_info=None):

        if 'filename' in split_info and  os.path.exists(split_info['filename']):
            print('RealisticDatasetDetrWrapper:: Load Dataset split: {} '.format(split_info['filename']))
        else:
            data_lists = {"train": [], "validation": [], "test": []}
            for name, split in {"train":training, "validation": validation, "test": testing}.items():
                split_idxs = split.indices
                for idx in split_idxs:
                    _, _, datanames = self.dataset.get_item_infos(idx)
                    data_lists[name].append(datanames)

            save_path = f"./data_configs/data_split.json"
            json.dump(data_lists, open(save_path, "w"), indent=2)
            split_info['filename'] = save_path

            print('RealisticDatasetDetrWrapper:: Save Dataset split: {} '.format(split_info['filename']))


    def get_real_test_ids(self, batch_size, fpose=False):

        if self.test is None:
            print(f"No Test set, Stop")
            return None
        training_idxs = self.training.indices
        test_idxs = self.test.indices
        training_poses, training_spec_fns = [], []
        for idx in training_idxs:
            pose_fbx, spec_fns, _ = self.dataset.get_item_infos(idx)
            training_poses.append(pose_fbx)
            training_spec_fns.extend(spec_fns)
        training_poses = set(training_poses)
        training_spec_fns = set(training_spec_fns)
        real_test_ids = []
        for idx in test_idxs:
            pose_fbx, spec_fns, _ = self.dataset.get_item_infos(idx)
            valid = True
            if fpose and pose_fbx in training_poses:
                valid = False
                continue
            for spec_fn in spec_fns:
                if spec_fn in training_spec_fns:
                    valid = False 
                    continue
            if valid:
                real_test_ids.append(idx)
        real_test = Subset(self.dataset, real_test_ids)
        print("{}::Real Test has total {} examples".format(self.__class__.__name__, len(real_test)))
        self.real_test = real_test
        self.loaders.real_test = DataLoader(self.real_test, self.batch_size)
        
    
    def save_to_wandb(self, experiment):
        """Save current data info to the wandb experiment"""
        # Split
        experiment.add_config('data_split', self.split_info)
        # save serialized split s.t. it's loaded to wandb
        split_datanames = {}
        split_datanames['training'] = [self.dataset.datapoints_names[idx] for idx in self.training.indices]
        split_datanames['validation'] = [self.dataset.datapoints_names[idx] for idx in self.validation.indices]
        split_datanames['test'] = [self.dataset.datapoints_names[idx] for idx in self.test.indices]
        with open(experiment.local_wandb_path() / 'data_split.json', 'w') as f_json:
            json.dump(split_datanames, f_json, indent=2, sort_keys=True)

        # data info
        self.dataset.save_to_wandb(experiment)
    
    # ---------- Standardinzation ----------------
    def standardize_data(self):
        """Apply data normalization based on stats from training set"""
        self.dataset.standardize(self.training)
        
    
    # --------- Managing predictions on this data ---------
    def predict(self, model, save_to, sections=['test'], single_batch=False, orig_folder_names=False, use_gt_stitches=False):
        """Save model predictions on the given dataset section"""
        prediction_path = os.path.join(save_to, ('nn_pred_' + datetime.now().strftime('%y%m%d-%H-%M-%S')))
        os.makedirs(prediction_path, exist_ok=True)
        model.module.eval()
        self.dataset.set_training(False)

        for section in sections:
            section_dir = prediction_path 
            os.makedirs(section_dir, exist_ok=True)
            cnt = 0
            with torch.no_grad():
                loader = self.get_loader(section)
                if loader:
                    for batch in loader:
                        cnt += 1
                        images = batch["image"].to(model.device_ids[0])
                        b = images.shape[0]
                        preds = model(images)

                        panel_shape = np.linalg.norm(preds["outlines"].cpu().detach().numpy().reshape((b, -1)) - batch["ground_truth"]["outlines"].cpu().detach().numpy().reshape(b, -1), axis=1)

                        self.dataset.save_prediction_batch(
                            preds, batch['img_fn'], batch['data_folder'], section_dir, 
                            model=model, orig_folder_names=orig_folder_names, images=images, use_gt_stitches=use_gt_stitches, panel_shape=panel_shape)

        return prediction_path
    

    def predict_single(self, model, image, dataname, save_to):
        device = model.device_ids[0] if hasattr(model, 'device_ids') and len(model.device_ids) > 0 else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        img_transform = transforms.make_image_transforms()
        img = img_transform(image).to(device).unsqueeze(0)
        output = model(img)
        panel_order, panel_idx, prediction_img = self.dataset.save_prediction_single(output, dataname, save_to)
        return panel_order, panel_idx, prediction_img
    
    def run_single_img(self, image, model, datawrapper):
        device = model.device_ids[0] if hasattr(model, 'device_ids') and len(model.device_ids) > 0 else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        img_transform = transforms.make_image_transforms()
        img = img_transform(image).to(device).unsqueeze(0)
        output = model(img), img
        return output, img

