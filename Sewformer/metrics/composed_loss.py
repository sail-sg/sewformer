import torch
import torch.nn as nn

import numpy as np

from entmax import SparsemaxLoss  # https://github.com/deep-spin/entmax

# My modules
from metrics.losses import *
from metrics.metrics import *
from sklearn.metrics import confusion_matrix


class ComposedLoss():
    """Base interface for compound loss objects"""

    def __init__(self, data_config, in_config={}):
        """
            Initialize loss components
            Accepts (in in_config):
            * Requested list of components
            * Additional configurations for losses (e.g. edge-origin agnostic evaluation)
            * data_stats -- for correct definition of losses
        """
        self.config = {  # defults
            'loss_components': [], 
            'quality_components': [],
        }
        self.config.update(in_config)  # override with requested settings

        self.with_quality_eval = True  # quality evaluation switch -- may allow to speed up the loss evaluation if False
        self.training = False  # training\evaluation state

        # Convenience properties
        self.l_components = self.config['loss_components']
        self.q_components = self.config['quality_components'] 

        if 'edge_pair_class' in self.l_components:
            self.bce_logits_loss = nn.BCEWithLogitsLoss()  # binary classification loss
        

    def __call__(self, preds, ground_truth, names=None, epoch=1000):
        """Evalute loss when predicting patterns.
            * Predictions are expected to follow the default GT structure, 
                but don't have to have all components -- as long as provided prediction is sufficient for
                evaluation of requested losses
            * default epoch is some large value to trigger stitch evaluation
            * Function returns True in third parameter at the moment of the loss stucture update
        """
        self.device = preds.device
        loss_dict = {}
        full_loss = 0.

        # match devices with prediction
        ground_truth = ground_truth.to(self.device)

        # ---- Losses ------
        main_losses, main_dict = self._main_losses(preds, ground_truth, None, epoch)
        full_loss += main_losses
        loss_dict.update(main_dict)

        # ---- Quality metrics  ----
        if self.with_quality_eval:
            with torch.no_grad():
                quality_breakdown = self._main_quality_metrics(preds, ground_truth, None, names)
                loss_dict.update(quality_breakdown)

        # final loss; breakdown for analysis; indication if the loss structure has changed on this evaluation
        return full_loss, loss_dict, False


    def eval(self):
        """ Loss to evaluation mode """
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def _main_losses(self, preds, ground_truth, gt_num_edges, epoch):
        """
            Main loss components. Evaluated in the same way regardless of the training stage
        """
        full_loss = 0.
        loss_dict = {}

        if 'edge_pair_class' in self.l_components:
            # flatten for correct computation
            pair_loss = self.bce_logits_loss(
                preds.view(-1), ground_truth.view(-1).type(torch.FloatTensor).to(self.device))
            loss_dict.update(edge_pair_class_loss=pair_loss)
            full_loss += pair_loss

        return full_loss, loss_dict

    def _main_quality_metrics(self, preds, ground_truth, gt_num_edges, names):
        """
            Evaluate quality components -- these are evaluated in the same way regardless of the training stage
        """
        loss_dict = {}
    
        if 'edge_pair_class' in self.q_components or 'edge_pair_stitch_recall' in self.q_components:
            edge_pair_class = torch.round(torch.sigmoid(preds))
            gt_mask = ground_truth.to(preds.device)

        if 'edge_pair_class' in self.q_components:
            acc = (edge_pair_class == gt_mask).sum().float() / gt_mask.numel()
            loss_dict.update(edge_pair_class_acc=acc)
        
        if 'edge_pair_stitch_recall' in self.q_components:
            prec, rec = self._prec_recall(edge_pair_class, gt_mask, target_label=1)
            loss_dict.update(stitch_precision=prec, stitch_recall=rec)

        return loss_dict

    def _prec_recall(self, preds, ground_truth, target_label):
        """ Evaluate precision/recall for given label in predictions """

        # correctly labeled as target label
        target_label_ids = (ground_truth == target_label).nonzero(as_tuple=True)
        correct_count = torch.count_nonzero(preds[target_label_ids] == target_label).float()

        # total number of labeled as target label
        pred_as_target_count = torch.count_nonzero(preds == target_label).float()

        # careful with division by zero
        precision = correct_count / pred_as_target_count if pred_as_target_count else 0
        recall = correct_count / len(target_label_ids[0]) if len(target_label_ids[0]) else 0

        return precision, recall


class ComposedPatternLoss():
    """
        Main (callable) class to define a loss on pattern prediction as composition of components
        NOTE: relies on the GT structure for pattern desctiption as defined in Pattern datasets 
    """
    def __init__(self, data_config, in_config={}):
        """
            Initialize loss components
            Accepts (in in_config):
            * Requested list of components
            * Additional configurations for losses (e.g. edge-origin agnostic evaluation)
            * data_stats -- for correct definition of losses
        """

        self.config = {  # defults
            'loss_components': ['shape'],  # 'loop',  
            'quality_components': [],  # 'loop',  
            'loop_loss_weight': 1.,
            'segm_loss_weight': 0.05,
            'stitch_tags_margin': 0.3,
            'epoch_with_stitches': 40, 
            'stitch_supervised_weight': 0.1,   # only used when explicit stitches are enabled
            'stitch_hardnet_version': False,

            'panel_origin_invariant_loss': True,
            'panel_order_inariant_loss': True,
            'order_by': 'placement',
            'epoch_with_order_matching': 0,
        }
        self.config.update(in_config)  # override with requested settings

        self.with_quality_eval = True  # quality evaluation switch -- may allow to speed up the loss evaluation if False
        self.training = False  # training\evaluation state
        self.debug_prints = False

        # Convenience properties
        self.l_components = self.config['loss_components']
        self.q_components = self.config['quality_components'] 

        self.max_panel_len = data_config['max_panel_len']
        self.max_pattern_size = data_config['max_pattern_len']

        data_stats = data_config['standardize']
        self.gt_outline_stats = {
            'shift': data_stats['gt_shift']['outlines'], 
            'scale': data_stats['gt_scale']['outlines']
        }

        # store moving-around-clusters info
        self.cluster_resolution_mapping = {}

        #  ----- Defining loss objects --------
        # NOTE I have to make a lot of 'ifs' as all losses have different function signatures
        # So, I couldn't come up with more consize defitions
        
        if 'shape' in self.l_components or 'rotation' in self.l_components or 'translation' in self.l_components:
            self.regression_loss = nn.MSELoss()  
        if 'loop' in self.l_components:
            self.loop_loss = PanelLoopLoss(self.max_panel_len, data_stats=self.gt_outline_stats)
        if 'stitch' in self.l_components:
            self.stitch_loss = PatternStitchLoss(
                self.config['stitch_tags_margin'], use_hardnet=self.config['stitch_hardnet_version'])
        if 'stitch_supervised' in self.l_components:
            self.stitch_loss_supervised = nn.MSELoss()
        if 'free_class' in self.l_components:
            self.bce_logits_loss = nn.BCEWithLogitsLoss()  # binary classification loss
        if 'segmentation' in self.l_components:
            # Segmenation output is Sparsemax scores (not SoftMax), hence using the appropriate loss
            self.segmentation = SparsemaxLoss()

        # -------- quality metrics ------
        if 'shape' in self.q_components:
            self.pattern_shape_quality = PanelVertsL2(self.max_panel_len, data_stats=self.gt_outline_stats)
        if 'discrete' in self.q_components:
            self.pattern_nums_quality = NumbersInPanelsAccuracies(
                self.max_panel_len, data_stats=self.gt_outline_stats)
        if 'rotation' in self.q_components:
            self.rotation_quality = UniversalL2(data_stats={
                'shift': data_stats['gt_shift']['rotations'], 
                'scale': data_stats['gt_scale']['rotations']}
            )
        if 'translation' in self.q_components:
            self.translation_quality = UniversalL2(data_stats={
                'shift': data_stats['gt_shift']['translations'], 
                'scale': data_stats['gt_scale']['translations']}
            )

    def __call__(self, preds, ground_truth, names=None, epoch=1000):
        """Evalute loss when predicting patterns.
            * Predictions are expected to follow the default GT structure, 
                but don't have to have all components -- as long as provided prediction is sufficient for
                evaluation of requested losses
            * default epoch is some large value to trigger stitch evaluation
            * Function returns True in third parameter at the moment of the loss stucture update
        """

        self.device = preds['outlines'].device
        self.epoch = epoch
        loss_dict = {}
        full_loss = 0.

        # match devices with prediction
        for key in ground_truth:
            ground_truth[key] = ground_truth[key].to(self.device)  

        # ------ GT pre-processing --------
        if self.config['panel_order_inariant_loss']:  # match panel order
            # NOTE: Not supported for 
            if 'segmentation' in self.l_components: 
                raise NotImplementedError('Order matching not supported for training with segmentation losses')
            gt_rotated = self._gt_order_match(preds, ground_truth) 
        else:  # keep original
            gt_rotated = ground_truth
        
        gt_num_edges = gt_rotated['num_edges'].int().view(-1)  # flatten

        if self.config['panel_origin_invariant_loss']:  # for origin-agnistic loss evaluation
            gt_rotated = self._rotate_gt(preds, gt_rotated, gt_num_edges, epoch)

        # ---- Losses ------
        main_losses, main_dict = self._main_losses(preds, gt_rotated, gt_num_edges)
        full_loss += main_losses
        loss_dict.update(main_dict)

        # stitch losses -- conditioned on the current process in training
        if epoch >= self.config['epoch_with_stitches'] and (
                'stitch' in self.l_components
                or 'stitch_supervised' in self.l_components
                or 'free_class' in self.l_components):
            losses, stitch_loss_dict = self._stitch_losses(preds, gt_rotated, gt_num_edges)
            full_loss += losses
            loss_dict.update(stitch_loss_dict)

        # ---- Quality metrics  ----
        if self.with_quality_eval:
            with torch.no_grad():
                quality_breakdown, corr_mask = self._main_quality_metrics(preds, gt_rotated, gt_num_edges, names)
                loss_dict.update(quality_breakdown)
                if "stitch_adj" in preds and "stitch_adj" in ground_truth:
                    b = ground_truth["stitch_adj"].shape[0]
                    # pred_adj = preds["stitch_adj"].max(1, keepdim=True)[1]
                    pred_adj = (torch.sigmoid(preds["stitch_adj"]) > 0.5).float()
                    gt_adj = ground_truth["stitch_adj"].view(b, -1)
                    cm_matrix = confusion_matrix(gt_adj.view(-1).cpu(), pred_adj.view(-1).cpu(), labels=[0, 1])
                    tp = cm_matrix[1][1]
                    tn = cm_matrix[0][0]
                    acc = 100.0 * tp / gt_adj.view(-1).sum().cpu().item()
                    loss_dict.update({"stitch_tp_acc": acc})
                    acc = 100.0 * tn / (gt_adj.view(-1).shape[0] - gt_adj.view(-1).sum().cpu().item())
                    loss_dict.update({"stitch_tn_acc": acc})

                    item_tps = 0
                    for i in range(b):
                        if pred_adj[i].eq(gt_adj[i]).sum().item() == pred_adj[i].shape[0]:
                            item_tps += 1
                    acc = 100.0 * item_tps / b
                    loss_dict.update({"stitch_batch_acc": acc})

                # stitches quality
                if epoch >= self.config['epoch_with_stitches']:
                    quality_breakdown = self._stitch_quality_metrics(
                        preds, gt_rotated, gt_num_edges, names, corr_mask)
                    loss_dict.update(quality_breakdown)

        loss_update_ind = (
            epoch == self.config['epoch_with_stitches'] and any((el in self.l_components for el in ['stitch', 'stitch_supervised', 'free_class']))
            or epoch == self.config['epoch_with_order_matching'] and self.config['panel_order_inariant_loss']) 

        # final loss; breakdown for analysis; indication if the loss structure has changed on this evaluation
        return full_loss, loss_dict, loss_update_ind

    def eval(self):
        """ Loss to evaluation mode """
        self.training = False

    def train(self, mode=True):
        self.training = mode

    
    def pred_to_aug_outlines(self, pred_outlines):

        def batch_control_to_abs_coord(starts, ends, control_scale):
            edge = ends - starts
            edge_perp = torch.cat([-edge[..., 1:], edge[..., :1]], dim=-1)
            control_start = starts + control_scale[..., :1] * edge
            control_point = control_start + control_scale[..., 1:] * edge_perp

            center_points = (1 - 0.7) ** 2 * starts + 2 * 0.7 * (1 - 0.7) * control_point + 0.7 ** 2 * ends
            return center_points
        
        scale = torch.tensor(self.gt_outline_stats["scale"]).to(pred_outlines.device)
        shift = torch.tensor(self.gt_outline_stats["shift"]).to(pred_outlines.device)

        de_outlines = pred_outlines * scale + shift
        b, n_panel, n_edge, _ = de_outlines.shape
        aug_points = torch.zeros((b, n_panel, 1, 2), dtype=de_outlines.dtype).to(de_outlines.device)
        aug_points = torch.cat((aug_points, de_outlines[..., :2]), dim=2)
        aug_points = torch.cumsum(aug_points, dim=2)
        num_points = aug_points.shape[2]

        starts = aug_points[:, :, :num_points-1, :]
        ends = aug_points[:, :, 1:num_points, :]
        control_scale = de_outlines[:, :, :, 2:]
        center_points = batch_control_to_abs_coord(starts, ends, control_scale)
        points = torch.cat((starts, center_points), dim=-1).reshape(b, n_panel, -1, 2)
        all_points = torch.cat((points, ends[:, :, -1:, :]), dim=2)

        num_all_points = all_points.shape[2]
        npstarts = np.concatenate([[i] * (num_all_points - 1 - i) for i in range(num_all_points)]).astype(np.int64)
        starts = torch.tensor(npstarts).unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(b, n_panel, -1, 2).to(pred_outlines.device)
        npends =  np.concatenate([[j for j in range(i, num_all_points)] for i in range(1, num_all_points)]).astype(np.int64)
        ends = torch.tensor(npends).unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(b, n_panel, -1, 2).to(pred_outlines.device)
        aug_edges = torch.gather(all_points, 2, ends) - torch.gather(all_points, 2, starts)

        aug_edges = (aug_edges - shift[:2]) / scale[:2]

        return aug_edges



    # ------- evaluation breakdown -------
    def _main_losses(self, preds, ground_truth, gt_num_edges):
        """
            Main loss components. Evaluated in the same way regardless of the training stage
        """
        full_loss = 0.
        loss_dict = {}

        if 'shape' in self.l_components:
            if "aug_outlines" in ground_truth:
                aug_pred_outlines = self.pred_to_aug_outlines(preds['outlines'])
                pattern_loss = self.regression_loss(aug_pred_outlines, ground_truth["aug_outlines"])
            else:
                pattern_loss = self.regression_loss(preds['outlines'], ground_truth['outlines'])
            
            full_loss += pattern_loss
            loss_dict.update(pattern_loss=pattern_loss)
            
        if 'loop' in self.l_components:
            loop_loss = self.loop_loss(preds['outlines'], gt_num_edges)
            full_loss += self.config['loop_loss_weight'] * loop_loss
            loss_dict.update(loop_loss=loop_loss)
            
        if 'rotation' in self.l_components:
            # independent from panel loop origin by design
            rot_loss = self.regression_loss(preds['rotations'], ground_truth['rotations'])
            full_loss += rot_loss
            loss_dict.update(rotation_loss=rot_loss)
        
        if 'translation' in self.l_components:
            # independent from panel loop origin by design
            translation_loss = self.regression_loss(preds['translations'], ground_truth['translations'])
            full_loss += translation_loss
            loss_dict.update(translation_loss=translation_loss)

        if 'segmentation' in self.l_components:

            pred_flat = preds['att_weights'].view(-1, preds['att_weights'].shape[-1])
            gt_flat = ground_truth['segmentation'].view(-1)

            # NOTE!!! SparseMax produces exact zeros
            segm_loss = self.segmentation(pred_flat, gt_flat)

            full_loss += self.config['segm_loss_weight'] * segm_loss
            loss_dict.update(segm_loss=segm_loss)

        return full_loss, loss_dict

    def _stitch_losses(self, preds, ground_truth, gt_num_edges):
        """
            Evaluate losses related to stitch info. Maybe calles or not depending on the training stage
        """
        full_loss = 0.
        loss_dict = {}

        if 'stitch' in self.l_components: 
            # Pushing stitch tags of the stitched edges together, and apart from all the other stitch tags
            stitch_loss, stitch_loss_breakdown = self.stitch_loss(
                preds['stitch_tags'], ground_truth['stitches'], ground_truth['num_stitches'])
            loss_dict.update(stitch_loss_breakdown)
            full_loss += stitch_loss
        
        if 'stitch_supervised' in self.l_components:
            stitch_sup_loss = self.stitch_loss_supervised(
                preds['stitch_tags'], ground_truth['stitch_tags'])      
            loss_dict.update(stitch_supervised_loss=stitch_sup_loss)
            full_loss += self.config['stitch_supervised_weight'] * stitch_sup_loss

        if 'free_class' in self.l_components:
            # free\stitches edges classification
            free_edges_loss = self.bce_logits_loss(
                preds['free_edges_mask'], ground_truth['free_edges_mask'].type(torch.FloatTensor).to(self.device))
            loss_dict.update(free_edges_loss=free_edges_loss)
            full_loss += free_edges_loss

        return full_loss, loss_dict

    def _main_quality_metrics(self, preds, ground_truth, gt_num_edges, names):
        """
            Evaluate quality components -- these are evaluated in the same way regardless of the training stage
        """
        loss_dict = {}

        correct_mask = None
        if 'discrete' in self.q_components:
            num_panels_acc, num_edges_acc, correct_mask, num_edges_correct_acc = self.pattern_nums_quality(
                preds['outlines'], gt_num_edges, ground_truth['num_panels'], pattern_names=names)
            loss_dict.update(
                num_panels_accuracy=num_panels_acc, 
                num_edges_accuracy=num_edges_acc,
                corr_num_edges_accuracy=num_edges_correct_acc)

        if 'shape' in self.q_components:
            shape_l2, correct_shape_l2 = self.pattern_shape_quality(
                preds['outlines'], ground_truth['outlines'], gt_num_edges, correct_mask)
            loss_dict.update(
                panel_shape_l2=shape_l2, 
                corr_panel_shape_l2=correct_shape_l2, 
            )
        
        if 'rotation' in self.q_components:
            rotation_l2, correct_rotation_l2 = self.rotation_quality(
                preds['rotations'], ground_truth['rotations'], correct_mask)
            loss_dict.update(rotation_l2=rotation_l2, corr_rotation_l2=correct_rotation_l2)

        if 'translation' in self.q_components:
            translation_l2, correct_translation_l2 = self.translation_quality(
                preds['translations'], ground_truth['translations'], correct_mask)
            loss_dict.update(translation_l2=translation_l2, corr_translation_l2=correct_translation_l2)
    
        return loss_dict, correct_mask

    def _stitch_quality_metrics(self, preds, ground_truth, gt_num_edges, names, correct_mask):
        """
            Quality components related to stitches prediction. May be called separately from main components 
            arrording to the training stage
        """
        loss_dict = {}
        if 'stitch' in self.q_components:
            stitch_prec, stitch_recall, corr_prec, corr_rec = self.stitch_quality(
                preds['stitch_tags'], preds['free_edges_mask'], 
                ground_truth['stitches'].type(torch.IntTensor).to(self.device), 
                ground_truth['num_stitches'],
                pattern_names=names, 
                correct_mask=correct_mask)
            loss_dict.update(
                stitch_precision=stitch_prec, 
                stitch_recall=stitch_recall,
                corr_stitch_precision=corr_prec, 
                corr_stitch_recall=corr_rec)
        
        if 'free_class' in self.q_components:
            free_class = torch.round(torch.sigmoid(preds['free_edges_mask']))
            gt_mask = ground_truth['free_edges_mask'].to(preds['free_edges_mask'].device)
            acc = (free_class == gt_mask).sum().float() / gt_mask.numel()

            loss_dict.update(free_edge_acc=acc)

        return loss_dict

    # ------ Ground truth panel order match -----
    def _gt_order_match(self, preds, ground_truth):
        """
            Find the permutation of panel in GT that is best matched with the prediction (by geometry)
            and return the GT object with all properties updated according to this permutation 
        """
        with torch.no_grad():
            gt_updated = {}

            # Match the order
            if self.config['order_by'] == 'placement':
                if ('translations' not in preds 
                        or 'rotations' not in preds):
                    raise ValueError('ComposedPatternLoss::Error::Ordering by placement requested but placement is not predicted')

                pred_feature = torch.cat([preds['translations'], preds['rotations']], dim=-1)
                gt_feature = torch.cat([ground_truth['translations'], ground_truth['rotations']], dim=-1)

            elif self.config['order_by'] == 'translation':
                if 'translations' not in preds:
                    raise ValueError('ComposedPatternLoss::Error::Ordering by translation requested but translation is not predicted')
                
                pred_feature = preds['translations']
                gt_feature = ground_truth['translations']
                
            elif self.config['order_by'] == 'shape_translation':
                if 'translations' not in preds:
                    raise ValueError('ComposedPatternLoss::Error::Ordering by translation requested but translation is not predicted')

                pred_outlines_flat = preds['outlines'].contiguous().view(preds['outlines'].shape[0], preds['outlines'].shape[1], -1)
                gt_outlines_flat = ground_truth['outlines'].contiguous().view(preds['outlines'].shape[0], preds['outlines'].shape[1], -1)

                pred_feature = torch.cat([preds['translations'], pred_outlines_flat], dim=-1)
                gt_feature = torch.cat([ground_truth['translations'], gt_outlines_flat], dim=-1)
 
            elif self.config['order_by'] == 'stitches':
                if ('free_edges_mask' not in preds
                        or 'translations' not in preds 
                        or 'rotations' not in preds):
                    raise ValueError('ComposedPatternLoss::Error::Ordering by stitches requested but free edges mask or placement are not predicted')
                
                pred_feature = torch.cat([preds['translations'], preds['rotations']], dim=-1)
                gt_feature = torch.cat([ground_truth['translations'], ground_truth['rotations']], dim=-1)

                if self.epoch >= self.config['epoch_with_stitches']: 
                    # add free mask as feature
                    # flatten per-edge info into single vector
                    # push preficted mask score to 0-to-1 range
                    pred_mask = torch.round(torch.sigmoid(preds['free_edges_mask'])).view(
                        preds['free_edges_mask'].shape[0], preds['free_edges_mask'].shape[1], -1)

                    gt_mask = ground_truth['free_edges_mask'].view(
                        ground_truth['free_edges_mask'].shape[0], ground_truth['free_edges_mask'].shape[1], -1)

                    pred_feature = torch.cat([pred_feature, pred_mask], dim=-1)
                    gt_feature = torch.cat([gt_feature, gt_mask], dim=-1)

                else:
                    print('ComposedPatternLoss::Warning::skipped order match by stitch tags as stitch loss is not enabled')      
                
            else:
                raise NotImplemented('ComposedPatternLoss::Error::Ordering by requested feature <{}> is not implemented'.format(
                    self.config['order_by']
                ))

            # run the optimal permutation eval
            gt_permutation = self._panel_order_match(pred_feature, gt_feature)

            # Update gt info according to the permutation
            gt_updated['outlines'] = self._feature_permute(ground_truth['outlines'], gt_permutation)
            gt_updated['num_edges'] = self._feature_permute(ground_truth['num_edges'], gt_permutation)
            gt_updated['empty_panels_mask'] = self._feature_permute(ground_truth['empty_panels_mask'], gt_permutation)
            
            # Not supported
            # gt_updated['segmentation'] = self._feature_permute(ground_truth['segmentation'], gt_permutation)

            if 'rotation' in self.l_components:
                gt_updated['rotations'] = self._feature_permute(ground_truth['rotations'], gt_permutation)
            if 'translation' in self.l_components:
                gt_updated['translations'] = self._feature_permute(ground_truth['translations'], gt_permutation)
                
            if self.epoch >= self.config['epoch_with_stitches'] and (
                    'stitch' in self.l_components
                    or 'stitch_supervised' in self.l_components
                    or 'free_class' in self.l_components):  # if there is any stitch-related evaluation

                gt_updated['stitches'] = self._stitch_after_permute( 
                    ground_truth['stitches'], ground_truth['num_stitches'], 
                    gt_permutation, self.max_panel_len
                )
                gt_updated['free_edges_mask'] = self._feature_permute(ground_truth['free_edges_mask'], gt_permutation)
                
                if 'stitch_supervised' in self.l_components:
                    gt_updated['stitch_tags'] = self._feature_permute(ground_truth['stitch_tags'], gt_permutation)

            # keep the references to the rest of the gt data as is
            for key in ground_truth:
                if key not in gt_updated:
                    gt_updated[key] = ground_truth[key]

        return gt_updated

    def _panel_order_match(self, pred_features, gt_features):
        """
            Find the best-matching permutation of gt panels to the predicted panels (in panel order)
            based on the provided panel features
        """
        with torch.no_grad():
            batch_size = pred_features.shape[0]
            pat_len = gt_features.shape[1]

            if self.epoch < self.config['epoch_with_order_matching']:
                # assign ordering randomly -- all the panel in the NN output have some non-zero signals at some point
                per_pattern_permutation = torch.stack(
                    [torch.randperm(pat_len, dtype=torch.long, device=pred_features.device) for _ in range(batch_size)]
                )
                return per_pattern_permutation

            # evaluate best order match
            # distances between panels (vectorized)
            total_dist_matrix = torch.cdist(
                pred_features.view(batch_size, pat_len, -1),   # flatten feature
                gt_features.view(batch_size, pat_len, -1))
            total_dist_flat_view = total_dist_matrix.view(batch_size, -1)

            # Assingment (vectorized in batch dimention)
            per_pattern_permutation = torch.full((batch_size, pat_len), fill_value=-1, dtype=torch.long, device=pred_features.device)
            for _ in range(pat_len):  # this many pair to arrange
                to_match_ids = total_dist_flat_view.argmin(dim=1)  # current global min is also a best match for the pair it's calculated for!
                
                rows = to_match_ids // total_dist_matrix.shape[1]
                cols = to_match_ids % total_dist_matrix.shape[1]

                for i in range(batch_size):  # only the easy operation is left unvectorized
                    per_pattern_permutation[i, rows[i]] = cols[i]
                    # exlude distances with matches
                    total_dist_matrix[i, rows[i], :] = float('inf')
                    total_dist_matrix[i, :, cols[i]] = float('inf')

            if torch.isfinite(total_dist_matrix).any():
                raise ValueError('ComposedPatternLoss::Error::Failed to match panel order')
        
        return per_pattern_permutation

    @staticmethod
    def _feature_permute(pattern_features, permutation):
        """
            Permute all given features (in the batch) according to given panel order permutation
        """
        with torch.no_grad():
            extended_permutation = permutation
            # match indexing with feature size
            if len(permutation.shape) < len(pattern_features.shape):
                for _ in range(len(pattern_features.shape) - len(permutation.shape)):
                    extended_permutation = extended_permutation.unsqueeze(-1)
                # expand just creates a new view without extra copies
                extended_permutation = extended_permutation.expand(pattern_features.shape)

            # collect features with correct permutation in pattern dimention
            indexed_features = torch.gather(pattern_features, dim=1, index=extended_permutation)
        
        return indexed_features

    @staticmethod
    def _stitch_after_permute(stitches, stitches_num, permutation, max_panel_len):
        """
            Update edges ids in stitch info after panel order permutation
        """
        with torch.no_grad():  # GT updates don't require gradient compute
            # add pattern dimention
            for pattern_id in range(len(stitches)):
                
                # inverse permutation for this pattern for faster access
                new_panel_ids_list = [-1] * permutation.shape[1]
                for i in range(permutation.shape[1]):
                    new_panel_ids_list[permutation[pattern_id][i]] = i

                # re-assign GT edge ids according to shift
                for side in (0, 1):
                    for i in range(stitches_num[pattern_id]):
                        edge_id = stitches[pattern_id][side][i]
                        panel_id = edge_id // max_panel_len
                        in_panel_edge_id = edge_id - (panel_id * max_panel_len)

                        # where is this panel placed
                        new_panel_id = new_panel_ids_list[panel_id]

                        # update with pattern-level edge id
                        stitches[pattern_id][side][i] = new_panel_id * max_panel_len + in_panel_edge_id
                
        return stitches

    # ------ Ground truth panel edge loop origin shift  ---------
    def _rotate_gt(self, preds, ground_truth, gt_num_edges, epoch):
        """
            Create a new GT object where panels are rotated to best match the predicted panels
        """
        with torch.no_grad():
            gt_updated = {}
            # for origin-agnistic loss evaluation
            gt_updated['outlines'], panel_leading_edges = self._batch_edge_order_match(
                preds['outlines'], ground_truth['outlines'], gt_num_edges)

            if epoch >= self.config['epoch_with_stitches'] and (
                    'stitch' in self.l_components
                    or 'stitch_supervised' in self.l_components
                    or 'free_class' in self.l_components):  # if there is any stitch-related evaluation
                gt_updated['stitches'] = self._gt_stitches_shift(
                    ground_truth['stitches'], ground_truth['num_stitches'], 
                    panel_leading_edges, gt_num_edges,
                    self.max_pattern_size, self.max_panel_len
                )
                gt_updated['free_edges_mask'] = self._per_panel_shift(
                    ground_truth['free_edges_mask'], 
                    panel_leading_edges, gt_num_edges)
                
                if 'stitch_supervised' in self.l_components:
                    gt_updated['stitch_tags'] = self._per_panel_shift(
                        ground_truth['stitch_tags'], panel_leading_edges, gt_num_edges)
            
            # keep the references to the rest of the gt data as is
            for key in ground_truth:
                if key not in gt_updated:
                    gt_updated[key] = ground_truth[key]

        return gt_updated

    @staticmethod
    def _batch_edge_order_match(predicted_panels, gt_panels, gt_num_edges):
        """
            Try different first edges of GT panels to find the one best matching with prediction
        """
        batch_size = predicted_panels.shape[0]
        if len(predicted_panels.shape) > 3:
            predicted_panels = predicted_panels.view(-1, predicted_panels.shape[-2], predicted_panels.shape[-1])
        if gt_panels is not None and len(gt_panels.shape) > 3:
            gt_panels = gt_panels.view(-1, gt_panels.shape[-2], gt_panels.shape[-1])
        
        chosen_panels = []
        leading_edges = []
        # choose the closest version of original panel for each predicted panel
        with torch.no_grad():
            for el_id in range(predicted_panels.shape[0]):
                num_edges = gt_num_edges[el_id]

                # Find loop origin with min distance to predicted panel
                chosen_panel, leading_edge, _ = ComposedPatternLoss._panel_egde_match(
                    predicted_panels[el_id], gt_panels[el_id], num_edges)

                # update choice
                chosen_panels.append(chosen_panel)
                leading_edges.append(leading_edge)

        chosen_panels = torch.stack(chosen_panels).to(predicted_panels.device)

        # reshape into pattern batch
        return chosen_panels.view(batch_size, -1, gt_panels.shape[-2], gt_panels.shape[-1]), leading_edges

    @staticmethod
    def _panel_egde_match(pred_panel, gt_panel, num_edges):
        """
            Find the optimal origin for gt panel that matches with the pred_panel best
        """
        shifted_gt_panel = gt_panel
        min_dist = ((pred_panel - shifted_gt_panel) ** 2).sum()
        chosen_panel = shifted_gt_panel
        leading_edge = 0
        for i in range(1, num_edges):  # will skip comparison if num_edges is 0 -- empty panels
            shifted_gt_panel = ComposedPatternLoss._rotate_edges(shifted_gt_panel, num_edges)
            dist = ((pred_panel - shifted_gt_panel) ** 2).sum()
            if dist < min_dist:
                min_dist = dist
                chosen_panel = shifted_gt_panel
                leading_edge = i
        
        return chosen_panel, leading_edge, min_dist

    @staticmethod
    def _per_panel_shift(panel_features, per_panel_leading_edges, panel_num_edges):
        """
            Shift given panel features accorging to the new edge loop orientations given
        """
        pattern_size = panel_features.shape[1]
        with torch.no_grad():
            for pattern_idx in range(len(panel_features)):
                for panel_idx in range(pattern_size):
                    edge_id = per_panel_leading_edges[pattern_idx * pattern_size + panel_idx] 
                    num_edges = panel_num_edges[pattern_idx * pattern_size + panel_idx]       
                    if num_edges < 3:  # just skip empty panels
                        continue
                    if edge_id:  # not zero -- shift needed. For empty panels its always zero
                        current_panel = panel_features[pattern_idx][panel_idx]
                        # requested edge goes into the first place
                        # padded area is left in place
                        panel_features[pattern_idx][panel_idx] = torch.cat(
                            (current_panel[edge_id:num_edges], current_panel[: edge_id], current_panel[num_edges:]))
        return panel_features

    @staticmethod
    def _gt_stitches_shift(
            gt_stitches, gt_stitches_nums, 
            per_panel_leading_edges, 
            gt_num_edges,
            max_num_panels, max_panel_len):
        """
            Re-number the edges in ground truth according to the perdiction-gt edges mapping indicated in per_panel_leading_edges
        """
        with torch.no_grad():  # GT updates don't require gradient compute
            # add pattern dimention
            for pattern_id in range(len(gt_stitches)):
                # re-assign GT edge ids according to shift
                for side in (0, 1):
                    for i in range(gt_stitches_nums[pattern_id]):
                        edge_id = gt_stitches[pattern_id][side][i]
                        panel_id = edge_id // max_panel_len
                        global_panel_id = pattern_id * max_num_panels + panel_id  # panel id in the batch
                        new_ledge = per_panel_leading_edges[global_panel_id]
                        panel_num_edges = gt_num_edges[global_panel_id]  # never references to empty (padding) panel->always positive number

                        inner_panel_id = edge_id - (panel_id * max_panel_len)  # edge id within panel
                        
                        # shift edge within panel
                        new_in_panel_id = inner_panel_id - new_ledge if inner_panel_id >= new_ledge else (
                            panel_num_edges - (new_ledge - inner_panel_id))
                        # update with pattern-level edge id
                        gt_stitches[pattern_id][side][i] = panel_id * max_panel_len + new_in_panel_id
                
        return gt_stitches

    @staticmethod
    def _rotate_edges(panel, num_edges):
        """
            Rotate the start of the loop to the next edge
        """
        panel = torch.cat((panel[1:num_edges], panel[0:1, :], panel[num_edges:]))

        return panel

