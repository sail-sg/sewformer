from pathlib import Path
import argparse
import yaml
import math
import numpy as np
import json
import shutil
import PIL
from PIL import Image
import requests

import torch
import torchvision.transforms as T
import torch.nn as nn


# My modules
import sys, os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pkg_path = "{}/SewFactory/packages".format(root_path)
print(pkg_path)
sys.path.insert(0, pkg_path)

import customconfig
import data
from data.transforms import denormalize_img_transforms
import models
from metrics.eval_detr_metrics import eval_detr_metrics
from experiment import ExperimentWrappper

def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', type=str, default='configs/test.yaml', help='YAML configuration file')
    parser.add_argument('--data-root', '-d', type=str, default='', help='Path to real images to be tested')
    parser.add_argument('--test-type', '-t', type=str, default='real', help='choice of evaluation type (sewfactory, deepfashion, real)')
    parser.add_argument('--save-root',  '-o', type=str, default='', help="output path for the predicted sewing pattern")


    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, args


## ======================== deepfashion ========================
class DeepFashionDataset(torch.utils.data.Dataset):
    def __init__(self, deep_fashion_root, resolution=1024):
        self._target_path = deep_fashion_root
        PIL.Image.init()
        self.resolution = resolution

        self._ref_image_fnames = [os.path.join(root, fname) for root, _dirs, files
                                  in os.walk(os.path.join(self._target_path)) for fname in files]
        self._ref_image_fnames = [(i, self.get_segm_path(i)) for i in self._ref_image_fnames]
        self._ref_image_fnames_has_mask = [i for i in self._ref_image_fnames if i[1] is not None]
        self._ref_image_fnames_has_mask_woman = [i for i in self._ref_image_fnames_has_mask if "WOMEN" in i[0]]
        print(len(self._ref_image_fnames), len(self._ref_image_fnames_has_mask), len(self._ref_image_fnames_has_mask_woman))
        
        self._img_transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor()
        ])
    
    def get_segm_path(self, path):
        _, path = os.path.split(path)
        img_id = os.path.splitext(path)[0]
        segm_path = os.path.join(self._target_path, 'segm', '{}_segm.png').format(img_id)
        if os.path.exists(segm_path):
            return segm_path
        return None
    
    def _get_source_appearance(self, idx):
        if len(self._ref_image_fnames_has_mask_woman) == 0:
            path = self._ref_image_fnames[idx % len(self._ref_image_fnames)]
            ref_img = self._load_raw_image(path[0])
            ref_img =np.array(ref_img) 
            ref_img_mask = np.array(Image.fromarray((np.ones(ref_img.shape[:2]) * 255).astype('uint8')))
        else:
            path = self._ref_image_fnames_has_mask_woman[idx]
            ref_img = self._load_raw_image(path[0])
            ref_img = np.array(ref_img)
            ref_img_mask = np.array(PIL.Image.open(path[1]))

        # remove background 
        ref_img = ref_img * (ref_img_mask > 0)[:, :, np.newaxis] + (ref_img_mask == 0)[:, :, np.newaxis] * 255
        ref_img = PIL.Image.fromarray(ref_img.astype(np.uint8))

        h, w = ref_img.size 
        min_size, max_size = min(h, w), max(h, w)

        pad_ref_img = T.Pad(padding=(int((max_size - h) / 2) , int((max_size - w) / 2)), fill=255)(ref_img)
        img_tensor = self._img_transform(pad_ref_img)

        return img_tensor, path[0]
    
    def __len__(self):
        if len(self._ref_image_fnames_has_mask_woman) == 0:
            return len(self._ref_image_fnames)
        return len(self._ref_image_fnames_has_mask_woman)
    
    def __getitem__(self, idx):
        img_tensor, dataname = self._get_source_appearance(idx)
        return img_tensor.unsqueeze(0), dataname

    
    def _load_raw_image(self, fname):
        image = PIL.Image.open(fname).convert('RGB')
        return image

def load_source_appearance(img_fn):
    ref_img = PIL.Image.open(img_fn).convert('RGB')
    h, w = ref_img.size
    min_size, max_size = min(h, w), max(h, w)
    pad_ref_img = T.Pad(padding=(int((max_size - h) / 2) , int((max_size - w) / 2)), fill=255)(ref_img)
    img_tensor = T.Compose([
                            T.Resize((384, 384)),
                            T.ToTensor()
                        ])(pad_ref_img)
    return img_tensor.unsqueeze(0), img_fn

def is_img_file(fn):
    img_endfix = ["png", "PNG", "jpg", "jpeg", "jpg", "JPEG", 'JPG']
    return fn.split('.')[-1] in img_endfix


if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True)
    system_info = customconfig.Properties('./system.json')

    config, args = get_values_from_args()

    shape_experiment = ExperimentWrappper(config, system_info['wandb_username'])  # finished experiment
    if not shape_experiment.is_finished():
        print('Warning::Evaluating unfinished experiment')
    
    # for detr based model
    if args.test_type != "sewfactory":
        shape_dataset, shape_datawrapper = shape_experiment.load_detr_dataset(
            [],   # assuming dataset root structure
            {'feature_caching': False, 'gt_caching':False},    # NOTE: one can change some data configuration for evaluation purposes here!
            unseen=True, batch_size=1)
    else:
        shape_dataset, shape_datawrapper = shape_experiment.load_detr_dataset(
            Path(system_info['datasets_path'])  if args.unseen else system_info['datasets_path'],   # assuming dataset root structure
            {'feature_caching': False, 'gt_caching':False},    # NOTE: one can change some data configuration for evaluation purposes here!
            unseen=args.unseen, batch_size=1)
    
    shape_model, criterion, device = shape_experiment.load_detr_model(shape_dataset.config, others=False)
    
    if args.test_type == "real":
        fns = [os.path.join(args.data_root, fn) for fn in os.listdir(args.data_root) if is_img_file(fn)]
        for idx, fn in enumerate(fns):
            img_tensor, img_fn = load_source_appearance(fn)
            dataname = os.path.basename(img_fn).split(".")[0]

            output = shape_model(img_tensor.to(device), return_stitches=True)
            _, _, prediction_img = shape_dataset.save_prediction_single(output, 
                                                                        dataname=dataname, 
                                                                        save_to=args.save_root, 
                                                                        return_stitches=False)
            shutil.copy2(img_fn, str(prediction_img.parent))
            print(f"end of #{idx}, {fn} ...")
        
    elif args.test_type == "deepfashion":
        deepfashion_dataset = DeepFashionDataset(deep_fashion_root=args.data_root)
        pred_save_root = f"{args.save_root}/{config['experiment']['run_name']}" 
        os.makedirs(pred_save_root, exist_ok=True)
        for idx in range(len(deepfashion_dataset)):
            img_tensor, dataname_path = deepfashion_dataset[idx]
            output = shape_model(img_tensor.to(device))
            dataname = os.path.basename(dataname_path).split(".")[0]
            save_to = os.path.join(pred_save_root)     
            _, _, prediction_img = shape_dataset.save_prediction_single(output, 
                                                                        dataname=dataname, 
                                                                        save_to=pred_save_root,
                                                                        return_stitches=False)
            shutil.copy2(dataname_path, os.path.dirname(prediction_img))
            print(f"end of #{idx}, {dataname_path} ...")
    elif args.test_type == "sewfactory":
        final_metrics, score_dict, _ = eval_detr_metrics(shape_model, criterion, shape_datawrapper, 0, "test")

    else:
        raise NotImplementedError("Not implemented test type: {}".format(args.test_type))


    
    


