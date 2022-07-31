import os
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

coco = COCO("dataset/annotations.json")
img_dir = "dataset/train/images"

dataset_fps = os.listdir(img_dir)
path = "dataset/train/masks/"
for img in list(coco.imgs.values()):
    if img["file_name"] in dataset_fps:
        file_name, _ = img["file_name"].split(".")
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        mask = np.zeros((img["height"], img["width"]), dtype="uint8")
        for i in range(len(anns)):
            mask += np.maximum(mask, coco.annToMask(anns[i]) * anns[i]["category_id"])
        img = Image.fromarray(mask)
        img.save(path + file_name + ".png")
        # plt.imsave(path + file_name + ".png", mask)

print("Máscaras creadas exitosamente")

