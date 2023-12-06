import torch
import torch.nn.functional as F
from torch import nn
import torchvision

import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report

from torch.nn import Sequential, Linear, ReLU

# from torch_geometric.nn import MessagePassing, DynamicEdgeConv, GATConv
# from torch_cluster import knn_graph

from .utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .position_encoding import build_position_encoding
from .transformer import build_transformer, TransformerDecoder, TransformerDecoderLayer

from metrics.composed_loss import ComposedLoss, ComposedPatternLoss

class GarmentDETRv6(nn.Module):
    def __init__(self, backbone, panel_transformer, num_panel_queries, num_edges, num_joints, **edge_kwargs):
        super().__init__()
        self.backbone = backbone
        
        self.num_panel_queries = num_panel_queries
        self.num_joint_queries = num_joints
        self.panel_transformer = panel_transformer

        self.hidden_dim = self.panel_transformer.d_model

        self.panel_embed = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)

        self.panel_joints_query_embed = nn.Embedding(self.num_panel_queries + self.num_joint_queries, self.hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.panel_rt_decoder = MLP(self.hidden_dim, self.hidden_dim, 7, 2)
        self.joints_decoder = MLP(self.hidden_dim, self.hidden_dim, 6, 2)

        self.num_edges = num_edges
        self.num_edge_queries = self.num_panel_queries * num_edges
        self.edge_kwargs = edge_kwargs["edge_kwargs"]


        self.panel_decoder = MLP(self.hidden_dim, self.hidden_dim, self.num_edges * 4, 2)
        self.edge_query_mlp = MLP(self.hidden_dim + 4, self.hidden_dim, self.hidden_dim, 1)

        self.build_edge_decoder(self.hidden_dim, self.edge_kwargs["nheads"], 
                                self.hidden_dim, self.edge_kwargs["dropout"], 
                                "relu", self.edge_kwargs["pre_norm"], 
                                self.edge_kwargs["dec_layers"])
        
        self.edge_embed = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)
        self.edge_cls = MLP(self.hidden_dim, self.hidden_dim // 2, 1, 2)
        self.edge_decoder = MLP(self.hidden_dim, self.hidden_dim, 4, 2)

    def build_edge_decoder(self, d_model, nhead, dim_feedforward, dropout, activation, normalize_before, num_layers):
        edge_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.edge_trans_decoder = TransformerDecoder(edge_decoder_layer, num_layers, decoder_norm, return_intermediate=True)
        self._reset_parameters()
    
    def _reset_parameters(self, ):
        for p in self.edge_trans_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, samples, gt_stitches=None, gt_edge_mask=None, return_stitches=False):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, panel_pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        B = src.shape[0]
        assert mask is not None
        panel_joint_hs, panel_memory, _ = self.panel_transformer(self.input_proj(src), mask, self.panel_joints_query_embed.weight, panel_pos[-1])
        panel_joint_hs = self.panel_embed(panel_joint_hs)
        panel_hs = panel_joint_hs[:, :, :self.num_panel_queries, :]
        joint_hs = panel_joint_hs[:, :, self.num_panel_queries:, :]
        output_panel_rt = self.panel_rt_decoder(panel_hs)

        output_rotations = output_panel_rt[:, :, :, :4]
        output_translations = output_panel_rt[:, :, :, 4:]

        out = {"rotations": output_rotations[-1], 
               "translations": output_translations[-1]}
        
        output_joints = self.joints_decoder(joint_hs)
        out.update({"smpl_joints": output_joints[-1]})

        edge_output = self.panel_decoder(panel_hs)[-1].view(B, self.num_panel_queries, self.num_edges, 4)
        edge_query = self.edge_query_mlp(torch.cat((panel_joint_hs[-1, :, :self.num_panel_queries, :].unsqueeze(2).expand(-1, -1, self.num_edges, -1), edge_output), dim=-1)).reshape(B, -1, self.hidden_dim).permute(1, 0, 2)

        tgt = torch.zeros_like(edge_query)
        memory = panel_memory.view(B, self.hidden_dim, -1).permute(2, 0, 1)         # latten NxCxHxW to HWxNxC
        edge_hs = self.edge_trans_decoder(tgt, memory, 
                                          memory_key_padding_mask=mask.flatten(1), 
                                          query_pos=edge_query).transpose(1, 2)
        
        output_edge_embed = self.edge_embed(edge_hs)[-1]

        output_edge_cls = self.edge_cls(output_edge_embed)
        output_edges = self.edge_decoder(output_edge_embed) + edge_output.view(B, -1, 4)


        out.update({"outlines": output_edges, "edge_cls": output_edge_cls})

        if return_stitches:
            pred_edge_prob = torch.sigmoid(output_edge_cls.squeeze(-1))

            if "max_num_edges" in self.edge_kwargs:
                num_edges = self.edge_kwargs["max_num_edges"]
            else:
                num_edges = (pred_edge_prob > 0.5).sum(dim=1).max().item()
                num_edges = num_edges + 1 if num_edges % 2 != 0 else num_edges
            
            stitch_edges = torch.argsort(pred_edge_prob, dim=1)[:, :num_edges]
            # use_gt = random.random() < self.edge_kwargs["gt_prob"] if "gt_prob" in self.edge_kwargs else False
            
            if gt_stitches is not None:

                stitch_edges = gt_stitches 
                edge_node_features = torch.gather(output_edge_embed, 1, stitch_edges.unsqueeze(-1).expand(-1, -1, output_edge_embed.shape[-1]).long())
                edge_node_features = edge_node_features.masked_fill(gt_edge_mask.unsqueeze(-1).expand(-1, -1, output_edge_embed.shape[-1]) == 0, 0)
            else:
                edge_node_features = torch.gather(output_edge_embed, 1, stitch_edges.unsqueeze(-1).expand(-1, -1, output_edge_embed.shape[-1]).long())
                reindex_stitches = None
        else:
            edge_node_features = output_edge_embed
        
        edge_norm_features = F.normalize(edge_node_features, dim=-1)
        edge_similarity = torch.bmm(edge_norm_features, edge_norm_features.transpose(1, 2))
        mask = torch.eye(edge_similarity.shape[1]).repeat(edge_similarity.shape[0], 1, 1).bool()
        edge_similarity[mask] = 0
        out.update({"edge_similarity": edge_similarity})
        
        return out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class StitchLoss():
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, similarity_matrix, gt_pos_neg_indices):
        simi_matrix = similarity_matrix.reshape(-1, similarity_matrix.shape[-1])
        tmp = simi_matrix[gt_pos_neg_indices[:, :, 0]]
        simi_res = torch.gather(tmp, -1, gt_pos_neg_indices[:, :, 1].unsqueeze(-1)) / 0.01
        ce_label = torch.zeros(simi_res.shape[0]).to(simi_res.device)
        return F.cross_entropy(simi_res.squeeze(-1), ce_label.long()), (torch.max(simi_res.squeeze(-1), 1)[1] == 0).sum() * 1.0 / simi_res.shape[0]

class StitchSimpleLoss():
    def __call__(self, similarity_matrix, gt_matrix, gt_free_mask=None):
        if gt_free_mask is not None:
            gt_free_mask = gt_free_mask.reshape(gt_free_mask.shape[0], -1)
            similarity_matrix = torch.masked_fill(similarity_matrix, gt_free_mask.unsqueeze(1), -float("inf"))
            similarity_matrix = torch.masked_fill(similarity_matrix, gt_free_mask.unsqueeze(-1), 0)

        simi_matrix = (similarity_matrix / 0.01).reshape(-1, similarity_matrix.shape[-1])
        gt = gt_matrix.reshape(-1, gt_matrix.shape[-1])
        gt_labels = torch.argmax(gt, dim=1).long()
        return F.nll_loss(F.log_softmax(simi_matrix, dim=-1), gt_labels), (torch.argmax(simi_matrix, dim=1) == gt_labels).sum() / simi_matrix.shape[0]



class SetCriterionWithOutMatcher(nn.Module):

    def __init__(self, data_config, in_config={}):
        super().__init__()
        self.config = {}
        self.config['loss'] = {
            'loss_components': ['shape', 'loop', 'rotation', 'translation'],  # , 'stitch', 'free_class'],
            'quality_components': ['shape', 'discrete', 'rotation', 'translation'],  #, 'stitch', 'free_class'],
            'panel_origin_invariant_loss': False,
            'loop_loss_weight': 1.,
            'stitch_tags_margin': 0.3,
            'epoch_with_stitches': 10000, 
            'stitch_supervised_weight': 0.1,   # only used when explicit stitch loss is used
            'stitch_hardnet_version': False,
            'panel_origin_invariant_loss': True
        }

        self.config['loss'].update(in_config)

        self.composed_loss = ComposedPatternLoss(data_config, self.config['loss'])

        self.stitch_cls_loss = nn.BCEWithLogitsLoss()
        self.stitch_ce_loss = StitchLoss()
        self.stitch_simi_loss = StitchSimpleLoss()
    
    def forward(self, outputs, ground_truth, names=None, epoch=1000):

        b, q = outputs["outlines"].shape[0], outputs["rotations"].shape[1]
        outputs["outlines"] = outputs["outlines"].view(b, q, -1, 4).contiguous()
        full_loss, loss_dict, _ = self.composed_loss(outputs, ground_truth, names, epoch)
        
        if "edge_cls" in outputs and 'lepoch' in self.config['loss'] and epoch >= self.config['loss']['lepoch']:
            if epoch == -1:
                st_edge_precision, st_edge_recall, st_edge_f1_score, st_precision, st_recall, st_f1_score, st_adj_precs, st_adj_recls, st_adj_f1s = self.prediction_stitch_rp(outputs, ground_truth)
                loss_dict.update({"st_edge_prec": st_edge_precision,
                                  "st_edge_recl": st_edge_recall,
                                  "st_edge_f1s": st_edge_f1_score,
                                  "st_prec": st_precision,
                                  "st_recl": st_recall,
                                  "st_f1s": st_f1_score,
                                  "st_adj_precs": st_adj_precs, 
                                  "st_adj_recls": st_adj_recls, 
                                  "st_adj_f1s": st_adj_f1s})
            edge_cls_gt = (~ground_truth["free_edges_mask"].view(b, -1)).float().to(outputs["edge_cls"].device)
            edge_cls_loss = torchvision.ops.sigmoid_focal_loss(outputs["edge_cls"].squeeze(-1), edge_cls_gt, reduction="mean")
            
            if self.config["loss"]["stitches"] == "ce" and epoch != -1:
                full_loss = full_loss * 10
                loss_dict.update({"stitch_cls_loss": 0.5 * edge_cls_loss})
                edge_cls_acc = ((F.sigmoid(outputs["edge_cls"].squeeze(-1)) > 0.5) == edge_cls_gt).sum().float() / (edge_cls_gt.shape[0] * edge_cls_gt.shape[1])
                loss_dict.update({"stitch_edge_cls_acc": edge_cls_acc})
                full_loss += loss_dict["stitch_cls_loss"]
                # ce loss
                stitch_loss, stitch_acc= self.stitch_ce_loss(outputs["edge_similarity"], ground_truth["label_indices"])
                if stitch_loss is not None and stitch_acc is not None:
                    loss_dict.update({"stitch_ce_loss": 0.01 * stitch_loss, "stitch_acc": stitch_acc})
                    full_loss += loss_dict["stitch_ce_loss"]
            elif self.config["loss"]["stitches"] == "simple" or epoch == -1:
                full_loss = full_loss * 5
                loss_dict.update({"stitch_cls_loss": 0.5 * edge_cls_loss})
                edge_cls_acc = ((F.sigmoid(outputs["edge_cls"].squeeze(-1)) > 0.5) == edge_cls_gt).sum().float() / (edge_cls_gt.shape[0] * edge_cls_gt.shape[1])
                loss_dict.update({"stitch_edge_cls_acc": edge_cls_acc})
                full_loss += loss_dict["stitch_cls_loss"]
                # simi loss
                stitch_loss, stitch_acc = self.stitch_simi_loss(outputs["edge_similarity"], ground_truth["stitch_adj"], ground_truth["free_edges_mask"])
                if stitch_loss is not None and stitch_acc is not None:
                    loss_dict.update({"stitch_ce_loss": 0.05 * stitch_loss, "stitch_acc": stitch_acc})
                    full_loss += loss_dict["stitch_ce_loss"]
            else:
                print("No Stitch Loss")
                stitch_loss, stitch_acc = None, None 

            if "smpl_joints" in ground_truth and "smpl_joints" in outputs:
                joints_loss = F.mse_loss(outputs["smpl_joints"], ground_truth["smpl_joints"])
                loss_dict.update({"smpl_joint_loss": joints_loss})
                full_loss += loss_dict["smpl_joint_loss"]
        
        return full_loss, loss_dict
    
    def prediction_stitch_rp(self, outputs, ground_truth):
        # only support batchsize=1

        if "edge_cls" in outputs:
            bs, q = outputs["outlines"].shape[0], outputs["rotations"].shape[1]
            st_edge_pres, st_edge_recls, st_edge_f1s, st_precs, st_recls, st_f1s, st_adj_precs, st_adj_recls, st_adj_f1s = [], [], [], [], [], [], [], [], []
            
            # import pdb; pdb.set_trace()

            for b in range(bs):
                edge_cls_gt = (~ground_truth["free_edges_mask"][b]).flatten()
                edge_cls_pr = (F.sigmoid(outputs["edge_cls"][b].squeeze(-1)) > 0.5).flatten()
                cls_rept = classification_report(edge_cls_gt.detach().cpu().numpy(), edge_cls_pr.detach().cpu().numpy(), labels=[0,1])
                strs = cls_rept.split("\n")[3].split()
                st_edge_precision, st_edge_recall, st_edge_f1_score = float(strs[1]), float(strs[2]), float(strs[3])
                edge_similarity = outputs["edge_similarity"][b]

                st_cls_pr = (F.sigmoid(outputs["edge_similarity"][b].squeeze(-1)) > 0.5).flatten()
                stitch_cls_rept = classification_report(st_cls_pr.detach().cpu().numpy(), ground_truth["stitch_adj"][b].flatten().detach().cpu().numpy(), labels=[0, 1])
                strs = stitch_cls_rept.split("\n")[3].split()
                st_adj_edge_precision, st_adj_edge_recall, st_adj_edge_f1_score = float(strs[1]), float(strs[2]), float(strs[3])

                st_adj_precs.append(st_adj_edge_precision)
                st_adj_recls.append(st_adj_edge_recall)
                st_adj_f1s.append(st_adj_edge_f1_score)

                if self.config["loss"]["stitches"] == "simple":
                    simi_matrix = edge_similarity + edge_similarity.transpose(0, 1)
                simi_matrix = torch.masked_fill(edge_similarity, (~edge_cls_pr).unsqueeze(0), -float("inf"))
                simi_matrix = torch.masked_fill(simi_matrix, (~edge_cls_pr).unsqueeze(-1), 0)
                simi_matrix = torch.triu(simi_matrix, diagonal=1)
                stitches = []
                num_stitches = edge_cls_pr.nonzero().shape[0] // 2
                for i in range(num_stitches):
                    index = (simi_matrix == torch.max(simi_matrix)).nonzero()
                    stitches.append((index[0, 0].cpu().item(), index[0, 1].cpu().item()))
                    simi_matrix[index[0, 0], :] = -float("inf")
                    simi_matrix[index[0, 1], :] = -float("inf")
                    simi_matrix[:, index[0, 0]] = -float("inf")
                    simi_matrix[:, index[0, 1]] = -float("inf")
                
                st_precision, st_recall, st_f1_score = SetCriterionWithOutMatcher.set_precision_recall(stitches, ground_truth["stitches"][b])

                st_edge_pres.append(st_edge_precision)
                st_edge_recls.append(st_edge_recall)
                st_edge_f1s.append(st_edge_f1_score)
                st_precs.append(st_precision)
                st_recls.append(st_recall)
                st_f1s.append(st_f1_score)
                # print(st_precision, st_recall, st_f1_score)

            return st_edge_pres, st_edge_recls, st_edge_f1s, st_precs, st_recls, st_f1s, st_adj_precs, st_adj_recls, st_adj_f1s
    
    @staticmethod
    def set_precision_recall(pred_stitches, gt_stitches):
        
        def elem_eq(a, b):
            return (a[0] == b[0] and a[1] == b[1]) or (a[0] == b[1] and a[1] == b[0])
        
        gt_stitches = gt_stitches.transpose(0, 1).cpu().detach().numpy()
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for pstitch in pred_stitches:
            for gstitch in gt_stitches:
                if elem_eq(pstitch, gstitch):
                    true_pos += 1
        false_pos = len(pred_stitches) - true_pos
        false_neg = len(gt_stitches) - (gt_stitches == -1).sum() / 2  - true_pos

        precision = true_pos / (true_pos + false_pos + 1e-6)
        recall = true_pos / (true_pos + false_neg)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        return precision, recall, f1_score
    
    def with_quality_eval(self, ):
        if hasattr(self.composed_loss, "with_quality_eval"):
            self.composed_loss.with_quality_eval = True
    
    def print_debug(self):
        self.composed_loss.debug_prints = True
    
    def train(self, mode=True):
        super().train(mode)
        self.composed_loss.train(mode)
    
    def eval(self):
        super().eval()
        if isinstance(self.composed_loss, object):
            self.composed_loss.eval()
            

def build(args):
    num_classes = args["dataset"]["max_pattern_len"]
    devices = torch.device(args["trainer"]["devices"][0] if isinstance(args["trainer"]["devices"], list) else args["trainer"]["devices"])
    backbone = build_backbone(args)
    panel_transformer = build_transformer(args)

    model = GarmentDETRv6(backbone, panel_transformer, num_classes, 14, 22, edge_kwargs=args["NN"])

    criterion = SetCriterionWithOutMatcher(args["dataset"], args["NN"]["loss"])
    criterion.to(devices)
    return model, criterion