from matplotlib import transforms
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tvtf

# from datasets.image_folder import ImageFolderDataset
from src.utils.getter import get_instance
from src. utils.device import move_to

from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import os
import csv

def crop_object(image, box):
    
    x_top_left = box[0]
    y_top_left = box[1]
    x_bottom_right = box[2]
    y_bottom_right = box[3]

    crop_img = image.crop((
        int(x_top_left), int(y_top_left),
        int(x_bottom_right), int(y_bottom_right)
    ))
    return crop_img

parser = argparse.ArgumentParser()
parser.add_argument('--cls_model', help='path to the classification model')
parser.add_argument('--det_model', help='path to the detection model')
parser.add_argument('--input', help='path to the images')
parser.add_argument('--output', help='path to the output directory')
parser.add_argument('--classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--thresh', type=float, 
                    help='score thresh test')
args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file(
    "./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.classes
cfg.MODEL.WEIGHTS = args.det_model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh

predictor = DefaultPredictor(cfg)


# Device
dev_id = 'cuda'
device = torch.device(dev_id)

# Load model
config = torch.load(args.cls_model, map_location=dev_id)
model = get_instance(config['config']['model']).to(device)
model.load_state_dict(config['model_state_dict'])

# Load data
transforms = tvtf.Compose([
    tvtf.Resize((224, 224)),
    tvtf.ToTensor(),
    tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
])

results = []

for i in os.listdir(args.input):
    filename = i.split('.')[0]
    
    img = Image.open(os.path.join(args.input, i))
    img_arr = np.array(img)
    outputs = predictor(img_arr)
    
    boxes = outputs['instances'].pred_boxes.tensor.detach().cpu()
    scores = outputs['instances'].scores.detach().cpu()

    idx = boxes[:, 0].argsort()
    boxes = boxes[idx]
    scores = scores[idx]
    
    with torch.no_grad():
        for i, (bbox, score) in enumerate(zip(boxes, scores)):
            if score > args.thresh:
                crop_img = transforms(crop_object(img, bbox))
                output = model(crop_img.unsqueeze(0).cuda())
                output = F.softmax(output, dim=1)
                _, preds = torch.max(output, dim=1)
                results.append([filename, preds.detach().cpu().tolist()[0]])


    os.makedirs(os.path.join(args.output, 'detection', filename), exist_ok=True)
    for i, (bbox, score) in enumerate(zip(boxes, scores)):
        crop_img = crop_object(img, bbox)
        if score > args.thresh:
            crop_img.save(os.path.join(args.output, 'detection', filename, f'{i}.png'))
    
    v = Visualizer(img_arr)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    os.makedirs(os.path.join(args.output, filename), exist_ok=True)
    Image.fromarray(v.get_image()).save(f'{args.output}/{filename}/{filename}.png')

with open(os.path.join(args.output, 'classification.txt'), 'w') as f:
    w = csv.writer(f)
    w.writerow(['filename', 'result'])
    w.writerows(results)




