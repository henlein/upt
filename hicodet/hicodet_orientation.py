"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import numpy as np
#import pocket
import datasets.transforms as T
from typing import Optional, List, Callable, Tuple
from torch.utils.data import Dataset
import torch
from pocket.data import ImageDataset, DataSubset


class HICODetOriSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]


class HICODetOri(ImageDataset):
    def __init__(self, root: str, anno_ori_file: str):
        super(HICODetOri, self).__init__(root)
        self.anno_ori_file = anno_ori_file
        with open(anno_ori_file, 'r') as f:
            anno_ori = json.load(f)

        self.num_ori_cls = 3
        self._load_annotation_and_metadata(anno_ori)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._anno)

    def __getitem__(self, i: int) -> tuple:
        anno = self._anno[i]

        fileid = [x for x in anno["filename"] if x.isdigit()]
        fileid_int = int(''.join(fileid[4:]))

        id_anno = {"fileid": torch.tensor(fileid_int), "boxes": anno["boxes"], "object": anno["object"], "labels": anno["labels"]}
        return self.load_image(os.path.join(self._root, self._anno[i]["filename"])), id_anno

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self.anno_ori_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    def split(self, ratio: float) -> Tuple[HICODetOriSubset, HICODetOriSubset]:
        perm = np.random.permutation(len(self._anno))
        n = int(len(perm) * ratio)
        return HICODetOriSubset(self, perm[:n]), HICODetOriSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._anno[idx]["filename"]

    def _ori_dict_to_vec(self, ori_dict):
        vector = [0, 0, 0, 0, 0, 0]
        keys = ori_dict.keys()
        if "n/a" in keys or len(keys) == 0:
            return vector
        elif "+x" in keys:
            vector[0] = 1
        elif "-x" in keys:
            vector[1] = 1
        elif "+y" in keys:
            vector[2] = 1
        elif "-y" in keys:
            vector[3] = 1
        elif "+z" in keys:
            vector[4] = 1
        elif "-z" in keys:
            vector[5] = 1
        else:
            print(ori_dict)
            print("!!!!!!!!!!!!!!!!!")
        return vector

    def calculate_pos_weights(self, class_counts, data_size):
        #https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
        pos_weights = np.ones_like(class_counts)
        neg_counts = [data_size - pos_count for pos_count in class_counts]
        for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
            pos_weights[cdx] = neg_count / (pos_count + 1e-5)
        return torch.as_tensor(pos_weights, dtype=torch.float, device='cuda')

    def _load_annotation_and_metadata(self, f: dict) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        self._anno = []
        self._filenames = []


        self.label2id = {
            "N/A": 0,
            "airplane": 5,
            "apple": 53,
            "backpack": 27,
            "banana": 52,
            "baseball bat": 39,
            "baseball glove": 40,
            "bear": 23,
            "bed": 65,
            "bench": 15,
            "bicycle": 2,
            "bird": 16,
            "blender": 83,
            "boat": 9,
            "book": 84,
            "bottle": 44,
            "bowl": 51,
            "broccoli": 56,
            "bus": 6,
            "cake": 61,
            "car": 3,
            "carrot": 57,
            "cat": 17,
            "cell phone": 77,
            "chair": 62,
            "clock": 85,
            "couch": 63,
            "cow": 21,
            "cup": 47,
            "desk": 69,
            "dining table": 67,
            "dog": 18,
            "donut": 60,
            "door": 71,
            "elephant": 22,
            "eye glasses": 30,
            "fire hydrant": 11,
            "fork": 48,
            "frisbee": 34,
            "giraffe": 25,
            "hair drier": 89,
            "handbag": 31,
            "hat": 26,
            "horse": 19,
            "hot dog": 58,
            "keyboard": 76,
            "kite": 38,
            "knife": 49,
            "laptop": 73,
            "microwave": 78,
            "mirror": 66,
            "motorcycle": 4,
            "mouse": 74,
            "orange": 55,
            "oven": 79,
            "parking meter": 14,
            "person": 1,
            "pizza": 59,
            "plate": 45,
            "potted plant": 64,
            "refrigerator": 82,
            "remote": 75,
            "sandwich": 54,
            "scissors": 87,
            "sheep": 20,
            "shoe": 29,
            "sink": 81,
            "skateboard": 41,
            "skis": 35,
            "snowboard": 36,
            "spoon": 50,
            "sports ball": 37,
            "stop sign": 13,
            "street sign": 12,
            "suitcase": 33,
            "surfboard": 42,
            "teddy bear": 88,
            "tennis racket": 43,
            "tie": 32,
            "toaster": 80,
            "toilet": 70,
            "toothbrush": 90,
            "traffic light": 10,
            "train": 7,
            "truck": 8,
            "tv": 72,
            "umbrella": 28,
            "vase": 86,
            "window": 68,
            "wine glass": 46,
            "zebra": 24
        }

        count_regions = 0
        count_labels = torch.zeros(6)
        zero_label_vec = torch.zeros(6)
        for anno_idx, (anno_key, anno) in enumerate(f["_via_img_metadata"].items()):
            bboxes = []
            fronts = []
            ups = []
            names = []
            labels = []
            for region in anno["regions"]:
                bbox = [region["shape_attributes"]["x"], region["shape_attributes"]["y"],
                        region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                        region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]

                front_vec = self._ori_dict_to_vec(region["region_attributes"]["front"])
                up_vec = self._ori_dict_to_vec(region["region_attributes"]["up"])

                if "category" not in region["region_attributes"]:
                    continue
                elif region["region_attributes"]["category"] == "human":
                    name = "person"
                elif region["region_attributes"]["category"] == "object":
                    name = region["region_attributes"]["obj name"]
                else:
                    print("???????????????????")
                    exit()
                if name not in self.label2id:
                    continue
                if all(v == 0 for v in front_vec):
                    continue


                bboxes.append(bbox)
                fronts.append(front_vec)
                ups.append(up_vec)
                names.append(self.label2id[name])
                labels.append(front_vec)# + up_vec)

            if len(bboxes) > 0:
                anno_dict = {"filename": anno["filename"], "boxes": torch.tensor(bboxes), "front": fronts, "up": ups, "object": torch.tensor(names), "labels": torch.FloatTensor(labels)}
                self._anno.append(anno_dict)
                for lab in anno_dict["labels"]:
                    count_labels += lab
                    count_regions += 1

        print(count_regions)
        print(count_labels)
        self.dataset_weights = self.calculate_pos_weights(count_labels, count_regions)
        print(self.dataset_weights)

class DataFactoryOri(Dataset):
    def __init__(self, dataset, name, train: bool):
        self.dataset = dataset

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if train:
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
        target['boxes'][:, :2] -= 1

        image, target = self.transforms(image, target)
        return image, target


if __name__ == "__main__":
    os.chdir("..")
    dataset = HICODetOri(
        root=os.path.join("D:/Corpora/HICO-DET", 'hico_20160224_det/images', "train2015"),
        anno_ori_file=os.path.join("D:/Corpora/HICO-DET", "via234_780 items_Dec 23.json")
    )
    #train_datset, test_dataset = dataset.split(0.8)
    #factory = DataFactoryOri(train_datset, "test", True)
    #print(factory[0])