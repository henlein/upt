"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import random
import warnings
import argparse
import wandb
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from hicodet.hicodet_orientation import HICODetOri, DataFactoryOri
from util import box_ops
from pocket.core import DistributedLearningEngine
import pocket
from upt import build_detector
from utils import custom_collate
from orientation_model import OrientationModel
warnings.filterwarnings("ignore")

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=1,
        rank=rank
    )
    print("::::::::::::")
    print(rank)
    wandb.init(project="ori_front_test", config=args)
    args = wandb.config
    print(args)

    # Fix seed
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    train_dataset = HICODetOri(
        root=os.path.join(args.data_root, 'hico_20160224_det/images', "train2015"),
        anno_ori_file=os.path.join(args.data_root, "orientation_annotation", "ALL_train.json"),
        #anno_ori_file = os.path.join(args.data_root, "orientation_annotation", "knife_train.json")
    )
    test_dataset = HICODetOri(
        root=os.path.join(args.data_root, 'hico_20160224_det/images', "train2015"),
        anno_ori_file=os.path.join(args.data_root, "orientation_annotation", "ALL_test.json")
        #anno_ori_file = os.path.join(args.data_root, "orientation_annotation", "knife_test.json")
    )

    trainset = DataFactoryOri(train_dataset, "train", True)
    testset = DataFactoryOri(test_dataset, "test", False)
    print("DataFactory created")
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset,
            num_replicas=args.world_size,
            rank=rank)
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )
    print("Loader created")

    args.human_idx = 1
    args.num_classes = 2

    label_weights = train_dataset.dataset_weights
    print(label_weights)
    orimodel = OrientationModel(args, label_weights)
    if rank == 0:
        wandb.watch(orimodel, log_freq=args.print_interval)

    print("CustomisedDLE")
    print(args.output_dir + "/" + wandb.run.name + "/")
    engine = OrientationDLE(
        orimodel, train_loader, test_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        #find_unused_parameters=True,
        cache_dir=args.output_dir+ "/" + wandb.run.name + "/"
        #cache_dir = args.output_dir + "/test/"
    )

    params = []
    for n, p in orimodel.named_parameters():
        if p.requires_grad:
            print(n)
            params.append(p)
    param_dicts = [{"params": params}]

    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    engine.save_checkpoint()
    engine(args.epochs)


class OrientationDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, test_loader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.test_loader = test_loader

    def _on_each_iteration(self):
        loss_dict, _ = self._state.net(
            *self._state.inputs, targets=self._state.targets)

        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        if self._rank == 0:
            wandb.log(loss_dict)
        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    def _on_end_epoch(self):
        if self._rank == 0:
            self.save_checkpoint()
            eval_results = self.test_orientation()
            print(eval_results)
            wandb.log(eval_results)

        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    def _on_end(self):
        if self._rank == 0:
            wandb.finish(0)
        #if dist.is_initialized():
        #    dist.j

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    @torch.no_grad()
    def test_orientation(self):
        net = self._state.net
        net.eval()

        correct_id = 0
        correct_baseline_id = 0
        correct_baseline_wo = 0
        found_id = 0
        all_all_pred = 0

        baseline_map = self._train_loader.dataset.dataset.map_front
        print(baseline_map)
        for batch_idx, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            inputs = pocket.ops.relocate_to_cuda(batch)
            targets = inputs[1]
            results = net(inputs[0])

            gold_boxes = targets[0]["boxes"]
            img_size = targets[0]["size"]
            gold_boxes = self.recover_boxes(gold_boxes, img_size)
            gold_boxes_ids = targets[0]["object"]
            gold_labels = targets[0]["labels"]


            pred_boxes = results["pred_boxes"][0]
            pred_boxes_ids = results["pred_boxes_label"][0]
            pred_labels = results["ffn_preds"][0]
            pred_labels = F.softmax(pred_labels, dim=1)

            for g_box, g_id, g_label in zip(gold_boxes, gold_boxes_ids, gold_labels):
                #if g_id.item() != 49:
                #    continue
                all_all_pred += 1

                possible_p_idx = []
                for p_idx, (p_box, p_id, p_label) in enumerate(zip(pred_boxes, pred_boxes_ids, pred_labels)):
                    box_iou = get_iou({"x1": p_box[0], "x2": p_box[2], "y1": p_box[1], "y2": p_box[3]},
                                           {"x1": g_box[0], "x2": g_box[2], "y1": g_box[1],
                                            "y2": g_box[3]})
                    if box_iou > 0.4 and p_id == g_id:
                        possible_p_idx.append(p_idx)

                if len(possible_p_idx) == 1:
                    found_id += 1
                    _, gold = torch.max(g_label, 0)
                    _, predicted = torch.max(pred_labels[possible_p_idx[0]], 0)
                    if gold.item() == predicted.item():
                        correct_id += 1

                    a = possible_p_idx[0]
                    b = pred_boxes_ids[a]
                    if b.item() in baseline_map:
                        c = baseline_map[b.item()]
                        d = torch.tensor(list(c))
                        _, predicted_baseline = torch.max(d, 0)
                        if gold.item() == predicted_baseline.item():
                            correct_baseline_id += 1

                    c = baseline_map["all"]
                    d = torch.tensor(list(c))
                    _, predicted_baseline = torch.max(d, 0)
                    if gold.item() == predicted_baseline.item():
                        correct_baseline_wo += 1

        #print(correct_id)
        #print(found_id)
        #print(correct_baseline_id)
        #print(all_all_pred)
        #print(correct_id / found_id)
        #print(correct_baseline_id / found_id)
        #print(correct_id / all_all_pred)
        results = {"found_correct_soft": correct_id / found_id, "found_correct_baseline": correct_baseline_id / found_id,
                   "all_correct_soft": correct_id / all_all_pred, "found_correct_baseline_wo": correct_baseline_wo / found_id}
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    #parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--data-root', default='../HicoDetDataset')
    #parser.add_argument('--data-root', default='D:/Corpora/HICO-DET')
    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=12, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=1, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--ff1-hidden', default=256, type=int)
    parser.add_argument('--input-version', default=1, type=int)  # 0=ModelNet, 1=Transformer, 2=Merge
    parser.add_argument('--loss-version', default=1, type=int)  # weight, stock

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    main(args=args, rank=0)
    #mp.spawn(main, nprocs=1, args=(args,))
