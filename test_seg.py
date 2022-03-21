import argparse
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image

import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='path to the segmentation model')
parser.add_argument('--input', help='path to the images')
parser.add_argument('--output', help='path to the output directory')
parser.add_argument('--classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--thresh', type=float, 
                    help='score thresh test')
args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file(
    "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.classes
cfg.MODEL.WEIGHTS = args.model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh

predictor = DefaultPredictor(cfg)

for i in os.listdir(args.input):
    filename = i.split('.')[0]

    img = Image.open(os.path.join(args.input, i))
    img = np.array(img)
    outputs = predictor(img)

    os.makedirs(args.output, exist_ok=True)

    visualize = Visualizer(img)
    visualize = visualize.draw_instance_predictions(outputs["instances"].to("cpu"))
    Image.fromarray(visualize.get_image()).save(os.path.join(args.output, f'{filename}.png'))
    