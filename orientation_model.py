from upt import build_detector
import torch
from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import box_iou, batched_nms
from ops import binary_focal_loss_with_logits
import torch.distributed as dist


class OrientationModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_classes = 12
        self.fg_iou_thresh = args.fg_iou_thresh
        self.alpha = args.alpha
        self.gamma = args.gamma

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

        self.ori_fc_1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.ori_fc_2 = nn.Linear(256, self.num_classes)

    def train(self, mode=True):
        super(OrientationModel, self).train(mode)
        self.upt.eval()

    def associate_with_ground_truth(self, pred_boxes, gold_boxes, targets, image_size):
        gt_bx = self.upt.recover_boxes(gold_boxes, image_size)
        x, y = torch.nonzero(box_iou(pred_boxes, gt_bx) >= self.fg_iou_thresh).unbind(1)
        labels = targets['labels'][y]
        #print("x", x)
        #print("y", y)
        #print(labels)
        return labels, x


    def forward(self, images: List[Tensor], targets: Optional[List[dict]] = None):
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)


        upt_feature = self.upt(images)
        #for x in upt_feature:
        #    x.pop("attn_maps")

        gold_boxes = [r['boxes'] for r in targets]

        pred_boxes = [r['boxes'] for r in upt_feature]
        pred_boxes_scores = [r['boxes_scores'] for r in upt_feature]
        pred_boxes_label = [r['boxes_labels'] for r in upt_feature]
        pbox_features = [r['unary_tokens'] for r in upt_feature]
        if self.training:
            target_labels = []
            ffn_preds = []
            for pbox, pbox_l, gbox, target, pbox_feature, image_size in zip(pred_boxes, pred_boxes_label, gold_boxes, targets, pbox_features, image_sizes):
                if pbox_feature.shape[0] == 0:
                    continue
                #print("==============")
                #print("pbox", pbox)
                #print("pbox_l", pbox_l)
                #print("pbox_features", pbox_feature)
                #print(str(pbox.shape) + " - " + str(pbox_feature.shape))
                assert pbox.shape[0] == pbox_feature.shape[0]
                #print("gbox", gbox)
                #print("target", target)
                #print("image_size", image_size)
                target_label, target_idx = self.associate_with_ground_truth(pbox, gbox, target, image_size)
                target_labels.append(target_label)
                filtered_pbox_feature = pbox_feature[target_idx]
                out1 = self.ori_fc_1(filtered_pbox_feature)
                out1 = self.relu1(out1)
                ffn_preds.append(self.ori_fc_2(out1))
                #print("---")

            target_labels = torch.cat(target_labels)
            ffn_preds = torch.cat(ffn_preds)

            n_p = ffn_preds.shape[0]
            if dist.is_initialized():
                world_size = dist.get_world_size()
                n_p = torch.as_tensor([n_p], device='cuda')
                dist.barrier()
                dist.all_reduce(n_p)
                n_p = (n_p / world_size).item()

            interaction_loss = binary_focal_loss_with_logits(
            torch.log(1 / torch.exp(-ffn_preds) + 1e-8),
                target_labels, reduction='sum', alpha=self.alpha, gamma=self.gamma)

            interaction_loss = interaction_loss / n_p

            loss_dict = dict(interaction_loss=interaction_loss)
            #print(loss_dict)
            return loss_dict

