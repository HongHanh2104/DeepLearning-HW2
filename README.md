# ICT2020 - Deep Learning course - Homework 2

## Requirements

```bash
conda install -c pytorch pytorch torchvision cudatoolkit
conda install -c anaconda pandas numpy pip pillow tqdm
pip install opencv-python
```

## Detectron2 Installation
```bash
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

## Convert dataset to COCO format

```bash
python mask2coco.py
```

## Train
#### Segmentation
```bash 
python train_detectron2.py --root_data './data' --method 'segmentation'
```
For training detection model, change the *segmentation* to *detection*.

#### Classification

```bash
python train.py --config .'./configs/train/chicken.yaml' --gpus 0
```

## Test
#### Segmentation

```bash
python test_seq.py --model '[path_to_trained_model]' --input './data/test' --thresh 0.9 --output seg_results
```

#### Classification
```bash
python test_cls.py --cls_model '[path_to_cls_model]' --det_model '[path_to_det_model]' --input './data/test/' --output cls_results --thresh 0.9
```
