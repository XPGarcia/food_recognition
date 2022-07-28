import os
from PIL import Image


def resize(img, new_width):
    width, height = img.size
    ratio = height / width
    new_height = int(ratio * new_width)
    resized_image = img.resize((new_width, new_height))
    return resized_image


DATA_DIR = "val/images/"
RESIZED_DIR = "val/images_resized_200/"
files = os.listdir(DATA_DIR)
extensions = ["jpg"]
for file in files:
    ext = file.split(".")[-1]
    if ext in extensions:
        image = Image.open(DATA_DIR + file)
        image_resized = resize(image, 200)
        file_path = f"{RESIZED_DIR}{file}"
        image_resized.save(file_path)
