import os
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from utils import get_classes

coco = COCO("dataset/annotations.json")
img_dir = "dataset/test/images_resized_256"

dataset_fps = os.listdir(img_dir)
path = "dataset/test/masks_resized_256/"
for img in list(coco.imgs.values()):
    if img["file_name"] in dataset_fps:
        file_name, _ = img["file_name"].split(".")
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        mask = np.zeros((img["height"], img["width"]), dtype="uint8")
        for i in range(len(anns)):
            cat = coco.cats[anns[i]["category_id"]]
            cat_index = get_classes().index(cat["name"])
            ann_mask = coco.annToMask(anns[i])
            ann_mask = ann_mask * cat_index
            mask += ann_mask
        img = Image.fromarray(mask)
        img.save(path + file_name + ".png")

print("Máscaras creadas exitosamente")
