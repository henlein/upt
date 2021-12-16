import torch
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import configparser
from utils import custom_collate, CustomisedDLE, DataFactory
from torch import nn
from util import box_ops
from tqdm import tqdm
import json

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


if __name__ == "__main__":
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    threshold = 0.9
    #trainset = DataFactory(name="hicodet", partition="train2015", data_root="D:/Corpora/HICO-DET/")
    testset = DataFactory(name="hicodet", partition="test2015", data_root="D:/Corpora/HICO-DET/")

    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')

    correct = []
    wrong_label = []
    wrong_det = []
    not_det = []
    results = {}
    for idx, (img, data) in tqdm(enumerate(testset), total=len(testset)):
        if idx < 4:
            continue
        #print( testset.dataset.filename(idx))
        boxes_h = data["boxes_h"]
        boxes_o = data["boxes_o"]
        o_label = data["object"]
        target_sizes = data["size"]

        boxes_stack = torch.cat((boxes_h, boxes_o), 0)
        h_label = torch.ones(len(boxes_h))
        labels_stack = torch.cat((h_label, o_label), 0)
        print("==============================")
        print(torch.unsqueeze(img, 0))

        features, pos = model.model.backbone(torch.unsqueeze(img, 0))
        print(features)
        print("....")
        print(pos)
        exit()
        model_outputs = model.model(torch.unsqueeze(img, 0))[0]
        print("======")
        print(model_outputs)
        exit()
        prob = nn.functional.softmax(model_outputs.logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.box_cxcywh_to_xyxy(model_outputs["pred_boxes"])
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(0)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
        boxes = boxes * scale_fct[None, :]


        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)][0]

        keep = results["scores"] > threshold
        scores = results["scores"][keep]
        pred_labels = results["labels"][keep]
        pred_boxes = results["boxes"][keep]
        print(scores)
        print(pred_labels)
        print(pred_boxes)
        exit()
        boxes_stack = box_ops.box_cxcywh_to_xyxy(boxes_stack)
        boxes_stack = boxes_stack * scale_fct[None, :]

        correct.append(0)
        wrong_label.append(0)
        wrong_det.append(0)


        for pbox, plab in zip(pred_boxes, pred_labels):
            found = False
            for tbox, tlab in zip(boxes_stack, labels_stack):
                hbox_overlap = get_iou(
                    {"x1": pbox[0].item(), "x2": pbox[2].item(), "y1": pbox[1].item(), "y2": pbox[3].item()},
                    {"x1": tbox[0].item(), "x2": tbox[2].item(), "y1": tbox[1].item(), "y2": tbox[3].item()})

                if hbox_overlap > 0.5:
                    found = True
                    if plab == tlab:
                        correct[-1] += 1
                    else:
                        wrong_label[-1] += 1
                    break

            if not found:
                wrong_det[-1] += 1
        not_det.append(len(labels_stack) - sum([correct[-1], wrong_label[-1]]))
        print(str(correct[-1]), str(wrong_label[-1]), str(wrong_det[-1]), str(not_det[-1]))
        print(sum(correct), sum(wrong_label), sum(wrong_det), sum(not_det))
        print("===========")
        exit()
    print(sum(correct))
    print(sum(wrong_label))
    print(sum(wrong_det))
    print(sum(not_det))
