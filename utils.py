"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import json
import os
import torch
import pickle
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

import sys
sys.path.append('detr')
import datasets.transforms as T


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

def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            if partition == "train2015":
                print("Use Train")
                self.dataset = HICODet(
                    root=os.path.join(data_root, 'hico_20160224_det/images', "merged2015"),
                    #anno_file=os.path.join(data_root, 'hicodet_obj_split',  'bicycle_train_2384.json'),
                    #anno_file=os.path.join(data_root, 'hicodet_verb_split', 'drive_train_1432.json'),
                    anno_file=os.path.join(data_root, 'hicodet_hoi_split', 'book_read_train_1827.json'),
                    target_transform=pocket.ops.ToTensor(input_format='dict')
                )
            elif partition == "test2015":
                print("Use Test")
                self.dataset = HICODet(
                    root=os.path.join(data_root, 'hico_20160224_det/images', "merged2015"),
                    anno_file=os.path.join(data_root, 'instances_test2015.json'),
                    #anno_file=os.path.join(data_root, 'hicodet_obj_split', 'bicycle_test_2384.json'),
                    #anno_file=os.path.join(data_root, 'hicodet_verb_split', 'drive_test_1432.json'),
                    #anno_file=os.path.join(data_root, 'hicodet_hoi_split', 'book_read_test_1827.json'),
                    target_transform=pocket.ops.ToTensor(input_format='dict')
                )
            else:
                print("ERROR2!!!")
                exit(0)
        else:
            print("ERROR!!!")
            exit(0)

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'): 
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
        ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        image, target = self.transforms(image, target)

        return image, target

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)

        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    @torch.no_grad()
    def test_hico(self, dataloader):
        net = self._state.net
        net.eval()

        all_correct = []
        verb_correct_obj_wrong = []
        obj_correct_verb_wrong = []
        obj_wrong_verb_wrong = []
        all_wrong = []
        missed = []

        debug_dict = {}

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        meter = DetectionAPMeter(
            91 * 2, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            target = batch[-1][0]
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            all_correct.append(0)
            verb_correct_obj_wrong.append(0)
            obj_correct_verb_wrong.append(0)
            obj_wrong_verb_wrong.append(0)
            all_wrong.append(0)

            # Skip images without detections
            if output is None or len(output) == 0:
                missed.append(len(target["verb"]))
                debug_dict[target["fileid"].item()] = (
                all_correct[-1], verb_correct_obj_wrong[-1], obj_correct_verb_wrong[-1], obj_wrong_verb_wrong[-1],
                all_wrong[-1], missed[-1])

                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            output.pop("attn_maps")

            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()

            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)


            inx = np.array([i for i in range(len(boxes_h)) if i % 2 == 0])
            boxes_h_filter = boxes_h[inx]
            boxes_o_filter = boxes_o[inx]
            objects_filter = objects[inx]
            scores_reshape = scores.reshape(-1, 2)

            pred_hbox = []
            pred_obox = []
            pred_obj = []
            pre_verb = []
            pred_verb_score = []
            for hbox, obox, score, obj in zip(boxes_h_filter, boxes_o_filter, scores_reshape, objects_filter.reshape(-1, 1)):
                if obj.item() not in [9, 6, 3, 7, 8]:
                    continue
                max_score, max_idx = torch.max(score, 0)
                if max_score.item() > 0.1:
                    pred_hbox.append(hbox)
                    pred_obox.append(obox)
                    pred_obj.append(obj.item())
                    pre_verb.append(max_idx.item())
                    pred_verb_score.append(score)

            if len(pre_verb) == 0:
                missed.append(len(target["verb"]))
                debug_dict[target["fileid"].item()] = (
                all_correct[-1], verb_correct_obj_wrong[-1], obj_correct_verb_wrong[-1], obj_wrong_verb_wrong[-1],
                all_wrong[-1], missed[-1])
                continue
                max_score, max_idx = torch.max(scores, 0)
                pred_hbox.append(boxes_h[max_idx])
                pred_obox.append(boxes_o[max_idx])
                pred_obj.append(objects[max_idx].item())
                pre_verb.append(max_idx.item() % 2)
                pred_verb_score.append(scores[max_idx])
            # Recover target box scale
            all_correct.append(0)
            verb_correct_obj_wrong.append(0)
            obj_correct_verb_wrong.append(0)
            obj_wrong_verb_wrong.append(0)
            all_wrong.append(0)
            for hbox, obox, verb, obj in zip(pred_hbox, pred_obox, pre_verb, pred_obj):
                found = False
                for ghbox, gobox, gverb, gobj in zip(gt_bx_h, gt_bx_o, target["verb"], target["object"]):
                    hbox_overlap = get_iou({"x1": hbox[0].item(), "x2": hbox[2].item(), "y1": hbox[1].item(), "y2": hbox[3].item()},
                                           {"x1": ghbox[0].item(), "x2": ghbox[2].item(), "y1": ghbox[1].item(),
                                            "y2": ghbox[3].item()})
                    obox_overlap = get_iou({"x1": obox[0].item(), "x2": obox[2].item(), "y1": obox[1].item(), "y2": obox[3].item()},
                                           {"x1": gobox[0].item(), "x2": gobox[2].item(), "y1": gobox[1].item(),
                                            "y2": gobox[3].item()})

                    if hbox_overlap > 0.5 and obox_overlap > 0.5:
                        found = True
                        if verb == gverb.item() and obj == gobj.item():
                            all_correct[-1] += 1
                        elif verb == gverb:
                            verb_correct_obj_wrong[-1] += 1
                        elif obj == gobj:
                            obj_correct_verb_wrong[-1] += 1
                        else:
                            obj_wrong_verb_wrong[-1] += 1
                        break
                if not found:
                    all_wrong[-1] += 1

            missed.append(len(target["verb"]) - sum([all_correct[-1], verb_correct_obj_wrong[-1], obj_correct_verb_wrong[-1], obj_wrong_verb_wrong[-1]]))
            debug_dict[target["fileid"].item()] = (all_correct[-1], verb_correct_obj_wrong[-1], obj_correct_verb_wrong[-1], obj_wrong_verb_wrong[-1], all_wrong[-1], missed[-1])
        return meter.eval(), {"all_correct": sum(all_correct), "verb_correct_obj_wrong": sum(verb_correct_obj_wrong), "obj_correct_verb_wrong": sum(obj_correct_verb_wrong),
                              "obj_wrong_verb_wrong": sum(obj_wrong_verb_wrong), "all_wrong": sum(all_wrong), "missed": sum(missed)}, debug_dict


    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
