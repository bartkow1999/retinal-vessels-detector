import tensorflow as tf
import os
from errno import EEXIST
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import UNetModel
import statisticalAnalysis


def mkdir(mypath):
    try:
        os.makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise


def superposition(img_list, masks_list, processed_img_list, iter_id):
    for i, (img, mask, processed_img) in enumerate(zip(img_list, masks_list, processed_img_list)):
        fig = plt.figure(figsize=(18, 8))

        fig.add_subplot(1, 3, 1)
        plt.imshow(img[:, :, :])
        plt.axis('off')

        fig.add_subplot(1, 3, 2)
        plt.imshow(processed_img, cmap='gray')
        plt.axis('off')

        fig.add_subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        statisticalAnalysis.confusion_matrix_data(f'IMG: {i}\n', img_true=mask[:, :, 0], img_predicted=processed_img)

        mkdir(f'resources/unet-results/try{iter_id}/')
        try:
            plt.savefig(f'resources/unet-results/try{iter_id}/{i}.png')
        except:
            continue


# PARAMETERS
####################
np.random.seed = 42

IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 3

TRAIN_PATH = 'resources/train/'
IMAGES_PATH = 'images/'
MASKS_PATH = 'masks/'



# TRAINING MODEL
####################

train_ids = next(os.walk(TRAIN_PATH + IMAGES_PATH))[2]

# loading training images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# print(X_train.shape)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
# print(Y_train.shape)

for i, img_file in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(TRAIN_PATH + IMAGES_PATH + img_file)[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[i] = img  # Fill empty X_train with values from img

    mask = imread(TRAIN_PATH + MASKS_PATH + img_file)[:, :, np.newaxis]
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Y_train[i] = mask

print('End of loading training images and masks!\n')

# initializing a neural network model
model = UNetModel.unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# model checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

# training
trained_model = model.fit(X_train, Y_train, validation_split=0.1, batch_size=2, epochs=100, callbacks=callbacks)
print('End of training\n')



# TESTING MODEL
####################

TEST_PATH = 'resources/test/'

test_ids = next(os.walk(TEST_PATH + IMAGES_PATH))[2]

# loading test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# print(X_test.shape)
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
# print(Y_test.shape)

for i, img_file in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(TEST_PATH + IMAGES_PATH + img_file)[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[i] = img  # Fill empty X_test with values from img

    mask = imread(TEST_PATH + MASKS_PATH + img_file)[:, :, np.newaxis]
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Y_test[i] = mask
print('End of loading test images and masks!\n')

# testing
predictions_test = model.predict(X_test, verbose=1)



# AFTER TESTING
####################

predictions_test_thresh = (predictions_test > 0.15).astype(np.uint8)

iter_id = 17 # number of algorithm iteration
superposition(X_test, Y_test, predictions_test_thresh, iter_id)
