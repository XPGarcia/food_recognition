import logging
import os

import matplotlib.pyplot as plt
import segmentation_models as sm

from tensorflow import keras
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from dataset import Dataset, Dataloder
from utils import get_preprocessing, get_training_augmentation, visualize

logging.getLogger("tensorflow").setLevel(logging.ERROR)

sm.set_framework("tf.keras")
sm.framework()

DATA_DIR = "dataset/"

x_train_dir = os.path.join(DATA_DIR, 'train/images_resized_256/')
y_train_dir = os.path.join(DATA_DIR, 'train/masks_resized_256/')

x_valid_dir = os.path.join(DATA_DIR, 'val/images_resized_256/')
y_valid_dir = os.path.join(DATA_DIR, 'val/masks_resized_256/')

# Define preprocessor

BACKBONE = "resnet34"  # vgg16, 'resnet18', inceptionv3,  resnet50
preprocess_input = sm.get_preprocessing(BACKBONE)

# Define classes
CLASSES = ["water", "onion", "avocado", "rice", "fish", "bread"]

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=CLASSES,
#    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input),
)

"""
# Testing the get_training_augmentation
image, mask = train_dataset[1]
visualize(
    image=image,
    water_mask=mask[..., 0].squeeze(),
    onion_mask=mask[..., 1].squeeze(),
    avocado_mask=mask[..., 2].squeeze(),
    rice_mask=mask[..., 3].squeeze(),
    fish_mask=mask[..., 4].squeeze(),
    bread_mask=mask[..., 5].squeeze(),
    background_mask=mask[..., 6].squeeze(),
    Total=mask[..., :-1].sum(axis=2).squeeze(),
)
"""

batch_size = 16

train_dataloader = Dataloder(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

model = sm.Unet(
    BACKBONE,
    encoder_weights='imagenet',
    classes=n_classes,
    activation=activation,
    encoder_freeze=True
)
model.summary()

# utils.plot_model(model, show_shapes=True)

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # learning_rate=0.001,
    loss=sm.losses.JaccardLoss(),  # 'jaccard_loss'
    metrics=["accuracy"]  # metrics.BinaryAccuracy()   m.metrics.IOUScore()
)

# print(train_dataloader[0][0].shape == (batch_size, 224, 224, 3))
# print(train_dataloader[0][1].shape == (batch_size, 224, 224, n_classes))

# train model
checkpoint = ModelCheckpoint(
    "models/food_recognition_model_resnet34_v2.keras",
    save_weights_only=True,
    save_best_only=True,
    # mode='min'
)
csv_logger = CSVLogger("models/food_recognition_model_resnet34_v2.log", separator=";", append=False)
reduceLR = ReduceLROnPlateau()

epochs = 10
history = model.fit(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=epochs,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
    callbacks=[checkpoint, reduceLR, csv_logger],
    verbose=1
)

history.history.keys()

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

# Plot training & validation iou_score values
# plt.figure(figsize=(30, 5))
plt.plot(epochs, accuracy, label="Training accuracy")
plt.plot(epochs, val_accuracy, label="Validation accuracy")
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training & validation iou_score values
# plt.figure(figsize=(30, 5))
plt.plot(epochs, loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
