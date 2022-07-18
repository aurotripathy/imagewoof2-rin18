import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
from rn18_model import ResNet18
import numpy as np
from keras.callbacks import ModelCheckpoint
import logging

logger = tf.get_logger()
logger.setLevel(logging.INFO)
logger.info(f'TF version:{tf.__version__} cuda version:{tf.sysconfig.get_build_info()["cuda_version"]}')

root_folder = '/home/auro/pyt-rn18-notebook'
train_folder = os.path.join(root_folder, "imagewoof2-320/train/")
val_folder = os.path.join(root_folder, "imagewoof2-320/val/")
image_size = (320, 320, 3)
target_size = (224, 224)
nb_epochs = 50
lr = 0.1
nb_classes = 10

imagegen = ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True,
                              zoom_range=[0.8, 1.2],
                              rotation_range=20,
                              horizontal_flip=True,)
imagegen.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B]
imagegen.std = 64.

# define train and val loaders
train = imagegen.flow_from_directory(train_folder, class_mode="categorical",
                                     shuffle=True, batch_size=128, target_size=target_size)
val = imagegen.flow_from_directory(val_folder, class_mode="categorical",
                                   shuffle=False, batch_size=128, target_size=target_size)

model = ResNet18(nb_classes)

def scheduler_1(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


def scheduler_2(epoch, lr):
    return 0.1 * tf.math.exp(-epoch / nb_epochs)
  
sgd_optimizer = tf.keras.optimizers.SGD(
  learning_rate=lr,
  momentum=0.9,
  nesterov=False,
  name='SGD',
)
  
adam_optimizer = keras.optimizers.Adam(learning_rate=lr, decay=1e-4)

lr_schedule_cb = tf.keras.callbacks.LearningRateScheduler(scheduler_2)

filepath = os.path.join('checkpoint',
                        'rn18-best-epoch-{epoch:02d}-acc-{val_accuracy:.3f}.hdf5')
checkpoint_cb = ModelCheckpoint(filepath=filepath,
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='max')

model.compile(loss='categorical_crossentropy',
              optimizer=sgd_optimizer,
              metrics=['accuracy'])

model.fit_generator(train, epochs=nb_epochs,
                    callbacks=[lr_schedule_cb, checkpoint_cb],
                    validation_data=val)
