import os
from tensorflow import keras
import numpy as np
import segmentation_models as sm
from dataset import Dataset, Dataloder
from utils import get_preprocessing
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics


DATA_DIR = "dataset/"

x_test_dir = os.path.join(DATA_DIR, 'test/images_resized_256')
y_test_dir = os.path.join(DATA_DIR, 'test/masks_resized_256')

CLASSES = ["water", "onion", "avocado", "rice", "fish", "bread"]

BACKBONE = "resnet34"  # vgg16, 'resnet18', inceptionv3,  resnet50
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

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # learning_rate=0.001,
    loss=sm.losses.JaccardLoss(),  # 'jaccard_loss'
    metrics=["accuracy"]  # metrics.BinaryAccuracy()   m.metrics.IOUScore()
)

# load best weights
model.load_weights('models/food_recognition_model_resnet34.keras')

scores = model.evaluate(test_dataloader)
print("Loss: {:.5}".format(scores[0]))
print("Accuracy: {:.5}".format(scores[1]))

"""## **Results**"""

n = len(test_dataset)
ids = np.random.choice(np.arange(len(test_dataset)), size=n)
ids = np.arange(n)

actual = []
predicted = []

for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)
    pr_mask = np.squeeze(pr_mask, axis=0)

    visualize(
        image=denormalize(image.squeeze()),
        gt_mask=gt_mask[..., :-1].sum(axis=2).squeeze(),
        pr_mask=pr_mask[..., :-1].sum(axis=2).squeeze(),
    )

#actual = numpy.random.binomial(1,.9,size = 1000)
#predicted = numpy.random.binomial(1,.9,size = 1000)
