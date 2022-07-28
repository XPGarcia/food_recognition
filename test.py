import os
import numpy as np
import segmentation_models as sm
from pycocotools.coco import COCO
from dataset import Dataset, Dataloder
from utils import visualize, denormalize, get_preprocessing, map_cats

DATA_DIR = ""

coco_test = COCO(DATA_DIR + "test/annotations.json")

x_test_dir = os.path.join(DATA_DIR, 'test/images')
y_test_dir = os.path.join(DATA_DIR, 'test/masks')

CLASSES = list(map(map_cats, list(coco_test.cats.values())))

BACKBONE = "resnet18"  # vgg16, 'resnet18', inceptionv3,  resnet50
preprocess_input = sm.get_preprocessing(BACKBONE)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input)
)
test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation, encoder_freeze=True)

# load best weights
model.load_weights('food_recognition_model.keras')

scores = model.evaluate(test_dataloader)
print("Loss: {:.5}".format(scores[0]))
print("Accuracy: {:.5}".format(scores[1]))

"""## **Results**"""

n = len(test_dataset)
# ids = np.random.choice(np.arange(len(test_dataset)), size=n)
ids = np.arange(n)

for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)

    visualize(
        image=denormalize(image.squeeze()),
        gt_mask=gt_mask.squeeze(),
        pr_mask=pr_mask.squeeze(),
    )
