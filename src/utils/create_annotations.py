from PIL import Image 
import numpy as np
from skimage import measure
from shapely.geometry import Polygon


def get_coco_json_format():
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }
    return coco_format

def create_annotation_format(image_id, category_id, annotation_id, polygon, segmentation):
    '''
    src: https://docs.trainingdata.io/v1.0/Export%20Format/COCO/
    '''
    
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area 

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }
    return annotation

def create_image_annotation(filename, width, height, image_id):
    images = {
        "file_name": filename,
        "height": height,
        "width": width,
        "id": image_id
    }
    return images

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
    return category_list

def create_sub_masks(mask_image, width, height):
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            pixel_str = str(mask_image.getpixel((x, y))[:3])
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
                sub_masks[pixel_str] = Image.new("1", (width + 2, height + 2))
            
            sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)
    
    return sub_masks

def create_submask_annotation(submask):
    contours = measure.find_contours(
        np.array(submask), 0.5, positive_orientation='low'
    )

    polygons = []
    segmentations = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        polygon = Polygon(contour)

        if polygon.is_empty:
            continue

        polygons.append(polygon)
        segmentation = np.array(polygon.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    return polygons, segmentations