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
from pocket.core import DistributedLearningEngine
import pocket
from upt import build_detector
from utils import custom_collate
from orientation_model import OrientationModel
warnings.filterwarnings("ignore")

def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=1,
        rank=rank
    )
    print("::::::::::::")
    print(rank)
    wandb.init(project="UPT-ORI Test", config=args)
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
        anno_ori_file=os.path.join(args.data_root, "orientation_annotation", "ALL_train.json")
    )
    test_dataset = HICODetOri(
        root=os.path.join(args.data_root, 'hico_20160224_det/images', "train2015"),
        anno_ori_file=os.path.join(args.data_root, "orientation_annotation", "ALL_test.json")
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

    @torch.no_grad()
    def test_orientation(self):
        net = self._state.net
        net.eval()

        all_correct = 0
        idx_correct = 0
        total_all = 0
        total_idx = 0

        all_correct_soft = 0

        total_loss = []
        for batch_idx, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            inputs = pocket.ops.relocate_to_cuda(batch)
            eval_loss, (pred_lab, gold_lab) = net(inputs[0], inputs[1])
            total_loss.append(eval_loss["interaction_loss"].item())
            print("....")
            print(gold_lab)
            pred_lab_soft = F.softmax(pred_lab, dim=1)
            pred_lab = pred_lab.sigmoid()
            print(pred_lab)
            print(pred_lab_soft)
            pred_lab = torch.round(pred_lab)
            print(pred_lab)
            for pred, gold in zip(pred_lab, gold_lab):
                if torch.equal(pred, gold):
                    all_correct += 1
                total_all += 1
                for pr_e, g_e in zip(pred, gold):
                    if torch.equal(pr_e, g_e):
                        idx_correct += 1
                    total_idx += 1
            _, predicted = torch.max(pred_lab_soft, 1)
            _, gold = torch.max(gold_lab, 1)
            print(predicted)
            print(gold)
            all_correct_soft += (predicted == gold).sum().item()

        results = {"eval_loss": sum(total_loss) / len(self.test_loader),
                   "all_correct": all_correct / total_all,
                   "all_correct_soft": all_correct_soft / total_all,
                   "idx_correct": idx_correct / total_idx}
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

    mp.spawn(main, nprocs=1, args=(args,))
