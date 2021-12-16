import os
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import pipeline

class MyDataset(Dataset):
    def __init__(self):
        image_path = "D:/Corpora/HICO-DET/hico_20160224_det/images/train2015"
        all_images = os.listdir(image_path)[:10]
        self.all_paths = [os.path.join(image_path, x) for x in all_images]

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, i):
        return self.all_paths[i]


if __name__ == "__main__":
    pipe = pipeline("object-detection", device=-1, model='facebook/detr-resnet-50')
    dataset = MyDataset()

    for out in tqdm(pipe(dataset, batch_size=2), total=len(dataset)):
        print(out)
        exit()