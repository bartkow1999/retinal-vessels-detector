import pathlib
import os
from errno import EEXIST

import cv2
import matplotlib.pyplot as plt

from imageProcessingFunctions import get_green_channel, make_normalizer, normalize_histogram, get_kernels, \
    morph_open_close, \
    subtract, thresh, create_mask, make_contours, make_bitwise
import statisticalAnalysis


def mkdir(mypath):
    try:
        os.makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise


def image_processing(img):
    img_green = get_green_channel(img)

    normalizer = make_normalizer()
    img_green_normalized = normalize_histogram(img_green, normalizer)

    kernels = get_kernels()
    img_after_morph = morph_open_close(img_green_normalized, kernels)

    img_vessels = subtract(img_after_morph, img_green_normalized)
    img_vessels_normalized = normalize_histogram(img_vessels, normalizer)
    img_vessels_normalized_thresh = thresh(img_vessels_normalized)

    noise_mask = create_mask(img_vessels_normalized_thresh)
    make_contours(img_vessels_normalized_thresh, noise_mask)
    img_vessels_corrected = make_bitwise(img_vessels_normalized, noise_mask)

    result = thresh(img_vessels_corrected)
    return result


def get_pictures(path):
    pictures = list()
    for file in pathlib.Path(path).iterdir():
        img = cv2.imread(str(file))
        pictures.append(img)
    return pictures


def superposition(img_list, masks_list, processed_img_list, iter_id):
    for i, (img, mask, processed_img) in enumerate(zip(img_list, masks_list, processed_img_list)):
        fig = plt.figure(figsize=(18, 8))

        fig.add_subplot(1, 3, 1)
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')

        fig.add_subplot(1, 3, 2)
        plt.imshow(processed_img, cmap='gray')
        plt.axis('off')

        fig.add_subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        statisticalAnalysis.confusion_matrix_data(f'IMG: {i}\n', img_true=mask[:, :, 0], img_predicted=processed_img)

        mkdir(f'resources/image-processing-results/try{iter_id}/')
        try:
            plt.savefig(f'resources/image-processing-results/try{iter_id}/{i}.png')
        except:
            continue


def main():
    iter_id = 2 # number of algorithm iteration
    TEST_PATH = 'resources/test/'

    img_list = get_pictures(TEST_PATH + 'images/')
    masks_list = get_pictures(TEST_PATH + 'masks/')
    processed_img_list = [image_processing(img) for img in img_list]

    superposition(
        img_list=img_list,
        masks_list=masks_list,
        processed_img_list=processed_img_list,
        iter_id=iter_id
    )


if __name__ == "__main__":
    main()
