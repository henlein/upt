from upt import build_detector
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from typing import Optional, List
from torchvision.ops.boxes import box_iou, batched_nms
from ops import binary_focal_loss_with_logits
import torch.distributed as dist


class OrientationModel(nn.Module):
    def __init__(self, args, label_weights=None):
        super().__init__()

        self.num_classes = 6
        self.fg_iou_thresh = args.fg_iou_thresh
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.input_type = args.input_version
        self.loss_type = args.loss_version
        self.label_weights = label_weights
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

        if self.input_type < 2:
            self.ori_fc_1 = nn.Linear(256, args.ff1_hidden)
        else:
            self.ori_fc_1 = nn.Linear(512, args.ff1_hidden)

        self.relu1 = nn.ReLU()
        self.ori_fc_2 = nn.Linear(args.ff1_hidden, self.num_classes)

    def train(self, mode=True):
        super(OrientationModel, self).train(mode)
        self.upt.eval()

    def associate_with_ground_truth(self, pred_boxes, gold_boxes, targets, image_size):
        gt_bx = self.upt.recover_boxes(gold_boxes, image_size)
        x, y = torch.nonzero(box_iou(pred_boxes, gt_bx) >= self.fg_iou_thresh).unbind(1)
        labels = targets['labels'][y]
        return labels, x

    def forward(self, images: List[Tensor], targets: Optional[List[dict]] = None):
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)


        upt_feature = self.upt(images)

        gold_boxes = [r['boxes'] for r in targets]

        pred_boxes = [r['boxes'] for r in upt_feature]
        pred_boxes_scores = [r['boxes_scores'] for r in upt_feature]
        pred_boxes_label = [r['boxes_labels'] for r in upt_feature]
        pbox_features = [r['unary_tokens'] for r in upt_feature]
        pbox_features_net = [r['boxes_hidden_states'] for r in upt_feature]
        if self.training or not self.training: #TODO: Für jetzt reicht es, aber nicht auf Dauer ...
            target_labels = []
            ffn_preds = []
            for pbox, pbox_l, gbox, target, pbox_feature, image_size, pbox_feature_net in \
                    zip(pred_boxes, pred_boxes_label, gold_boxes, targets, pbox_features, image_sizes, pbox_features_net):
                if pbox_feature.shape[0] == 0:
                    continue
                assert pbox.shape[0] == pbox_feature.shape[0]
                assert pbox_feature_net.shape[0] == pbox_feature.shape[0]

                target_label, target_idx = self.associate_with_ground_truth(pbox, gbox, target, image_size)
                target_labels.append(target_label)
                filtered_pbox_feature = pbox_feature[target_idx]
                filtered_pbox_net_feature = pbox_feature_net[target_idx]

                input_feature = None
                if self.input_type == 0:
                    input_feature = filtered_pbox_net_feature
                elif self.input_type == 1:
                    input_feature = filtered_pbox_feature
                elif self.input_type == 2:
                    input_feature = torch.cat((filtered_pbox_feature, filtered_pbox_net_feature), dim=1)
                else:
                    print("Not supportet Model Input Type")
                    exit()

                out1 = self.ori_fc_1(input_feature)
                out1 = self.relu1(out1)
                ffn_preds.append(self.ori_fc_2(out1))

            target_labels = torch.cat(target_labels)
            ffn_preds = torch.cat(ffn_preds)

            n_p = ffn_preds.shape[0]
            if dist.is_initialized():
                world_size = dist.get_world_size()
                n_p = torch.as_tensor([n_p], device='cuda')
                dist.barrier()
                dist.all_reduce(n_p)
                n_p = (n_p / world_size).item()

            if self.loss_type == 0:
                interaction_loss = binary_cross_entropy_with_logits(ffn_preds, target_labels,
                                                                reduction='sum', weight=self.label_weights)
            elif self.loss_type == 1:
                interaction_loss = binary_focal_loss_with_logits(
                   torch.log(1 / torch.exp(-ffn_preds) + 1e-8), target_labels, reduction='sum', alpha=self.alpha, gamma=self.gamma)
            else:
                print("Not supportet Loss Type")
                exit()

            #interaction_loss = binary_focal_loss_with_logits(
            #   torch.log(1 / torch.exp(-ffn_preds) + 1e-8), target_labels, reduction='sum', alpha=self.alpha, gamma=self.gamma)

            interaction_loss = interaction_loss / n_p

            loss_dict = dict(interaction_loss=interaction_loss)
            return loss_dict, (ffn_preds, target_labels)

