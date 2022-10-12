# UPT: Unary&ndash;Pairwise Transformers
## Main Differences to UPT
### Model
AffordanceUPT can be found in upt.py 
Changes worth mentioning:
* Supporting two classes: Gibsonian and Telic
* Modularization to support pretrained Huggingface DETR / other object detection models.

### Data Loader
DataLoader for annotated HicoDet Data can be found in hicodet/hicodet.py
Adapted to support Gibsonian / telic annotations from "HOI.csv"
The DataLoader gets initiated in: utils.py:70 ....
This is important for later usage (see Usage).

## Usage
### Hyperparameter / Arguments
Hyperparameters were found via wandb.
The old default hyperparameters of upt were commented out.
The main hyperparameters:
* data-root: Path for HicoDet Data. See "HicoDet Dataset" for more information
* output-dir: where to save the trained model.

### HicoDet Dataset
The files under "data-root" should look like this (in this folder are more files then needed. It growed historically). : 
<img src="assets/dataset_Structure.png" align="justify" width="500">
* hico_20160224-det: the original dataset
* All important annotated files.
* I will prepare something under the folder: prepared_dataset

### Train
* python main.py (change "data-root" and "output-dir")

### Evaluation
* CUDA_VISIBLE_DEVICES=1 python main.py --world-size 1 --eval --resume "path_to_model.pt"

### Notes
* utils.py:87 let you change the evaluations for different objectsets. but it need also a bit more preperation. -> for maby later.
* wandb maby needs a little bit of cleaning ...

