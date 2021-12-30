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
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from hicodet.hicodet_orientation import HICODetOri, DataFactoryOri
from upt import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory
from orientation_model import OrientationModel
warnings.filterwarnings("ignore")

def main(rank, args):

    # Fix seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    dataset = HICODetOri(
        root=os.path.join(args.data_root, 'hico_20160224_det/images', "train2015"),
        anno_ori_file=os.path.join(args.data_root, "via234_780 items_Dec 23.json")
    )
    train_dataset, test_dataset = dataset.split(0.8)

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
    args.num_classes = 3

    orimodel = OrientationModel(args)

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
    max_norm = args.clip_max_norm
    for _ in range(20):
        for batch in train_loader:
            inputs = batch[:-1]
            targets = batch[-1]
            loss_dict = orimodel(*inputs, targets=targets)
            print(loss_dict)
            if loss_dict['interaction_loss'].isnan():
                raise ValueError(f"The HOI loss is NaN")

            loss = sum(loss for loss in loss_dict.values())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(orimodel.parameters(), max_norm)
            optim.step()
        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
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
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    main(0, args)
