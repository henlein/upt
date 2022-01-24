from upt import build_detector
import numpy as np
import pocket
import torch
from torch import nn, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from typing import Optional, List
from torchvision.ops.boxes import box_iou, batched_nms
from ops import binary_focal_loss_with_logits
import torch.distributed as dist
from interaction_head import ModifiedEncoder

class OrientationModel(nn.Module):
    def __init__(self, args, label_weights=None):
        super().__init__()

        self.num_classes = 3
        self.fg_iou_thresh = args.fg_iou_thresh
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.input_type = args.input_version
        self.loss_type = args.loss_version
        self.label_weights = label_weights
        self.ff1_hidden = args.ff1_hidden
        class_corr = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        with torch.no_grad():
            self.upt = build_detector(args, class_corr)
            checkpoint = torch.load("checkpoints/ckpt_41940_20.pt", map_location='cpu')
            self.upt.load_state_dict(checkpoint['model_state_dict'])
            self.upt.eval()

        for param in self.upt.parameters():
            param.requires_grad = False

        self.comp_layer = pocket.models.TransformerEncoderLayer(
            hidden_size=1024,
            return_weights=True
        )
        if self.input_type == 1:
            pretrained_comp = self.upt.interaction_head.comp_layer.state_dict()
            self.comp_layer.load_state_dict(pretrained_comp)
        '''
        self.coop_layer = ModifiedEncoder(
            hidden_size=args.hidden_dim,
            representation_size=args.repr_dim,
            num_layers=2,
            return_weights=True
        )

        print("::::::::::::::::::::::::::::::::::::::")
        pretrained_coop = self.upt.interaction_head.coop_layer.state_dict()
        print(pretrained_coop)
        self.coop_layer.load_state_dict(pretrained_coop)
        print("Loading worked :)")
        '''
        self.ori_fc_1 = nn.Linear(1024, self.ff1_hidden)

        self.relu1 = nn.ReLU()
        self.ori_fc_2 = nn.Linear(self.ff1_hidden, self.num_classes)

    def train(self, mode=True):
        super(OrientationModel, self).train(mode)
        self.upt.eval()

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        gt_bx_h = self.upt.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.upt.recover_boxes(targets['boxes_o'], targets['size'])

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)

        labels = targets['labels'][y]
        return labels, x


    def forward(self, images: List[Tensor], targets: Optional[List[dict]] = None):
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)


        upt_feature = self.upt(images)

        boxes = upt_feature["boxes"]
        bh = upt_feature["bh"]
        bo = upt_feature["bo"]
        logits = upt_feature["logits"]
        prior = upt_feature["prior"]
        objects = upt_feature["objects"]
        pairwise_tokens = upt_feature["pairwise_tokens"]
        pre_pairwise_tokens = upt_feature["pre_pairwise_tokens"]
        if self.training:
            filtered_labels = []
            filtered_features = []
            for bx, h, o, target in zip(boxes, bh, bo, targets):
                lab, x = self.associate_with_ground_truth(bx[h], bx[o], target)
                filtered_labels.append(lab)
                filtered_feature = pre_pairwise_tokens[x] # HERE Change Input Feature
                filtered_features.append(filtered_feature)
            filtered_labels = torch.cat(filtered_labels)
            filtered_features = torch.cat(filtered_features)
        else:
            filtered_features = pairwise_tokens

        out, _ = self.comp_layer(filtered_features)

        out = self.ori_fc_1(out)
        out = self.relu1(out)
        out = self.ori_fc_2(out)

        if not self.training:
            results = {"h_box": boxes[0][bh[0]], "o_box": boxes[0][bo[0]], "pred_boxes_label": objects[0],
                       "image_sizes": image_sizes[0], "ffn_preds": out}
            return results

        n_p = out.shape[0]
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        if self.loss_type == 0:
            interaction_loss = binary_cross_entropy_with_logits(out, filtered_labels,
                                                                reduction='sum', weight=self.label_weights)
        elif self.loss_type == 1:
            interaction_loss = binary_focal_loss_with_logits(
                torch.log(1 / torch.exp(-out) + 1e-8), filtered_labels, reduction='sum', alpha=self.alpha,
                gamma=self.gamma)
        else:
            print("Not supportet Loss Type")
            exit()

        interaction_loss = interaction_loss / n_p

        loss_dict = dict(interaction_loss=interaction_loss)
        return loss_dict, (out, filtered_labels)