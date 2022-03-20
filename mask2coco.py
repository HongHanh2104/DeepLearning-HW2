import json
import os
import glob
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

from src.utils.create_annotations import *

category_ids = {
    "background": 0,
    "chicken": 1
}

category_colors = {
    "(0, 0, 0)": 0,
    "(255, 255, 255)": 1
}


def images_annotations_info(maskpath):
    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)

    annotation_id = 0
    image_id = 0
    annotations = []
    images = []

    for mask_image in tqdm(glob.glob(maskpath + "/*.jpg")):
        
        original_file_name = os.path.basename(
            mask_image).split(".")[0] + ".jpg"

        mask_image_open = Image.open(mask_image).convert("P").convert("RGB")
        w, h = mask_image_open.size

        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            category_id = category_colors[color]
            if category_id == 0:
                continue

            polygons, segmentations = create_submask_annotation(sub_mask)

            if category_id == 1:
                multi_poly = MultiPolygon(polygons)

                annotation = create_annotation_format(
                    multi_poly, segmentations, image_id, category_id, annotation_id)

                annotations.append(annotation)
                annotation_id += 1
            else:
                for i in range(len(polygons)):
                    
                    segmentation = [
                        np.array(polygons[i].exterior.coords).ravel().tolist()]

                    annotation = create_annotation_format(
                        polygons[i], segmentation, image_id, category_id, annotation_id)

                    annotations.append(annotation)
                    annotation_id += 1
        image_id += 1

    return images, annotations, annotation_id

if __name__ == "__main__":
    mask_path = './data/train/mask'
    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)

    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(
        mask_path
    )

    with open("./data/train.json", "w") as f:
        json.dump(coco_format, f)

    print("Create annotations for images.")