import json
import os
import sys
import glob
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

from src.utils.create_annotations import create_annotation_format, create_category_annotation, create_image_annotation, create_sub_masks, create_submask_annotation, get_coco_json_format 

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

    for mask in tqdm(glob.glob(maskpath + "/*.jpg")):
        original_filename = os.path.basename(mask).split(".")[0] + ".jpg"

        mask_img = Image.open(mask).convert("P").convert("RGB")
        w, h = mask_img.size 

        image = create_image_annotation(original_filename, w, h, image_id)
        images.append(image)

        submasks = create_sub_masks(mask_img, w, h)
        for color, submask in submasks.items():
            category_id = category_colors[color]
            if category_id == 0:
                continue

            polygons, segmentations = create_submask_annotation(submask)

            if category_id == 1:
                multi_poly = MultiPolygon(polygons)

                annotation = create_annotation_format(
                    image_id, category_id, annotation_id, multi_poly, segmentations
                )

                annotations.append(annotation)
                annotation_id += 1
            else:
                for i in range(len(polygons)):
                    segmentation = [
                        np.array(polygons[i].exterior.coords).ravel().tolist()
                    ]

                    annotation = create_annotation_format(
                        image_id, category_id, annotation_id, polygons[i], segmentation
                    )

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