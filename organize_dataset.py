import os, shutil, pathlib, errno
from pycocotools.coco import COCO


def make_subset(original_dir, new_base_dir, subset_name, folders, images, start_index, end_index):
    for category in folders:
        src_dir = original_dir / category
        dir = new_base_dir / subset_name / category
        try:
            os.makedirs(dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        for i in range(start_index, end_index):
            shutil.copyfile(src=src_dir / images[i]["file_name"], dst=dir / images[i]["file_name"])
            os.remove(src_dir / images[i]["file_name"])


DATA_DIR = ""

coco_train = COCO(DATA_DIR + "train/annotations.json")
coco_test = COCO(DATA_DIR + "test/annotations.json")
coco_val = COCO(DATA_DIR + "val/annotations.json")

original_dir = pathlib.Path(DATA_DIR + "train")
new_base_dir = pathlib.Path(DATA_DIR)

folders = ["images", "masks"]
total = len(coco_train.imgs)
itest = int(total * 0.9)

images = list(coco_train.imgs.values())

make_subset(original_dir, new_base_dir, "test", folders, images, start_index=itest, end_index=total)
print("test subset created successfully!")