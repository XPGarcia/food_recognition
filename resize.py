import os
from PIL import Image


def resize(img, new_width):
    # width, height = img.size
    # ratio = height / width
    # new_height = int(ratio * new_width)
    # resized_image = img.resize((new_width, new_height))
    resized_image = img.resize((256, 256))
    return resized_image


DATA_DIR = "dataset/test/masks/"
RESIZED_DIR = "dataset/test/masks_resized_256/"
files = os.listdir(DATA_DIR)
extensions = ["jpg", "png"]
for file in files:
    ext = file.split(".")[-1]
    if ext in extensions:
        image = Image.open(DATA_DIR + file)
        image_resized = resize(image, 256)
        file_path = f"{RESIZED_DIR}{file}"
        image_resized.save(file_path)
