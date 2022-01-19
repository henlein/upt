from upt import build_detector
import numpy as np
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
        if self.input_type < 2:
            self.ori_fc_1 = nn.Linear(256, args.ff1_hidden)
        else:
            self.ori_fc_1 = nn.Linear(512, args.ff1_hidden)

        self.relu1 = nn.ReLU()
        self.ori_fc_2 = nn.Linear(args.ff1_hidden, self.num_classes)


    def train(self, mode=True):
        super(OrientationModel, self).train(mode)
        self.upt.eval()

    def compute_orientation_loss(self, boxes, bh, bo, logits, prior, targets):
        labels = torch.cat([
            self.upt.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        print("labels", labels)
        print("labels", labels.size())
        exit()
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
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)


        upt_feature = self.upt(images)

        boxes = upt_feature["boxes"]
        bh = upt_feature["bh"]
        bo = upt_feature["bo"]
        logits = upt_feature["logits"]
        prior = upt_feature["prior"]
        pairwise_tokens = upt_feature["pairwise_tokens"]
        labels = torch.cat([
            self.upt.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])


        print("labels", labels)
        print("labels", labels.size())
        exit()
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)