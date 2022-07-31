import os, shutil, pathlib, errno


def to_png(file_name):
    file_name, _ = file_name.split(".")
    return file_name + ".png"


def make_subset(original_dir, new_base_dir, subset_name, folders, images, start_index, end_index):
    for category in folders:
        if category == "masks":
            images = list(map(to_png, images))
        src_dir = original_dir / category
        dir = new_base_dir / subset_name / category
        try:
            os.makedirs(dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        for i in range(start_index, end_index):
            shutil.copyfile(src=src_dir / images[i], dst=dir / images[i])
            # os.remove(src_dir / images[i]["file_name"])


DATA_DIR = "dataset"

original_dir = pathlib.Path(DATA_DIR)
new_base_dir = pathlib.Path(DATA_DIR)

images = os.listdir(DATA_DIR + "/images")

folders = ["images", "masks"]
total = len(images)
ival = int(total * 0.8)
itest = int(total * 0.9)

make_subset(original_dir, new_base_dir, "train", folders, images, start_index=0, end_index=ival)
make_subset(original_dir, new_base_dir, "val", folders, images, start_index=ival, end_index=itest)
make_subset(original_dir, new_base_dir, "test", folders, images, start_index=itest, end_index=total)
