import keras
from DataLoader import Dataloder
from Dataset import Dataset
from LoadingUtils import visualize
from modifiedAugmentation import get_training_augmentation
import segmentation_models as sm
import numpy as np

# %% Train
BACKBONE = 'resnet34'
BATCH_SIZE = 3
CLASSES = ['specularity', 'saturation', 'artifact', 'blur', 'contrast', 'bubbles', 'instrument', 'blood']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 9  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

trainDataset = Dataset()
# image, mask = trainDataset[20]
# visualize(
#     image=image,
#     specularity=mask[..., 0].squeeze(),
#     saturation=mask[..., 1].squeeze(),
#     artifact=mask[..., 2].squeeze(),
#     blur=mask[..., 3].squeeze(),
#     contrast=mask[..., 4].squeeze(),
#     bubbles=mask[..., 5].squeeze(),
#     instrument=mask[..., 6].squeeze(),
#     blood=mask[..., 7].squeeze(),
# )


train_dataloader = Dataloder(trainDataset, batch_size=BATCH_SIZE, shuffle=True)

callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=train_dataloader,
)
