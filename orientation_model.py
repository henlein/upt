import torch.nn as nn
from upt import build_detector
import torch
from torch import nn, Tensor
from typing import Optional, List


class OrientationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
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
            checkpoint = torch.load("checkpoints/upt-14.pt", map_location='cpu')
            self.upt.load_state_dict(checkpoint['model_state_dict'])
            self.upt.eval()

        for param in self.upt.parameters():
            param.requires_grad = False

        self.front_fc = nn.Linear(256, 6)
        self.up_fc = nn.Linear(256, 6)


    def associate_with_ground_truth(self, pred_boxes, gold_boxes, size):
        n = pred_boxes.shape[0]
        labels = torch.zeros(n, self.num_classes, device=pred_boxes.device)

        gt_bx = self.upt.recover_boxes(gold_boxes, size)

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets['labels'][y]] = 1

        return labels


    def compute_ori_loss(self, pred_boxes, gold_boxes, pred_front, pred_up, targets):
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]
        prior = prior[x, y]
        labels = labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )

        return loss / n_p

    def forward(self, images: List[Tensor], targets: Optional[List[dict]] = None):
        print("================")
        print(images)
        print(targets)
        print("================")

        upt = self.upt(images)
        print(upt)
        #rep = torch.tensor([x["bhs"] for x in upt])
        pred_front = torch.cat([self.front_fc(rep) for rep in upt["bhs"]])
        pred_up = torch.cat([self.up_fc(rep) for rep in upt["bhs"]])

        pred_boxes = [r['boxes'] for r in upt]
        gold_boxes = [r['boxes'] for r in targets]
        if self.training:
            interaction_loss = self.compute_ori_loss(pred_boxes, gold_boxes, pred_front, pred_up, targets)
            loss_dict = dict(
                interaction_loss=interaction_loss
            )
            return loss_dict
        print(upt)
        exit()
        pass