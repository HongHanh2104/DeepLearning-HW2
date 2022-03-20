import pprint
import argparse

import yaml
import os
import datetime

from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg

'''
Follow this tutorial: https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/
'''
def train(json_file,
          image_path,
          method='segmentation'):
    
    # Register the chicken dataset to tell detectron2 
    # how to obtain the dataset
    
    register_coco_instances("chicken", {}, "data/train.json", "data/train/img")

    # Fine-tune a coco-pretrained R50-FPN Mask RCNN model 
    # on the chicken dataset

    cfg = get_cfg()
    if method == 'segmentation':
        cfg.merge_from_file(
            "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    elif method == 'detection':
        cfg.merge_from_file(
            "detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    
    cfg.DATASETS.TRAIN = ("chicken",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 10

    if method == 'segmentation':
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    elif method == 'detection':
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

    cfg.SOLVER.IMS_PER_BATCH = 12
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = (
        500
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    
    train_id = "chicken" + '-' + method
    train_id += '-' + datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    if method == 'segmentation':
        cfg.OUTPUT_DIR = 'checkpoints/segmentation/' + train_id + '/'
    elif method == 'detection':
        cfg.OUTPUT_DIR = 'checkpoints/detection/' + train_id + '/'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data')
    parser.add_argument('--json', default='train.json')
    parser.add_argument('--image', default='train/img')
    parser.add_argument('--method')

    args = parser.parse_args()

    json_file = os.path.join(args.root_data, args.json)
    img_path = os.path.join(args.root_data, args.image)

    train(json_file, img_path, args.method)
 