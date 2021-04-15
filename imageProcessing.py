import pathlib
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_green_channel(img):
    return img[:, :, 1]


def normalize_histogram(img, normalizer):
    return normalizer.apply(img)


def get_kernels():
    kernel_sizes = [(5, 5), (7, 7), (15, 15), (21, 21)]
    results = list()
    for size in kernel_sizes:
        results.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size))
    return results


def morph_open(img, kernels):
    img = img.copy()
    for kernel in kernels:
        img = cv2.erode(img, kernel)
        img = cv2.dilate(img, kernel)
    return img


def morph_close(img, kernels):
    img = img.copy()
    for kernel in kernels:
        img = cv2.dilate(img, kernel)
        img = cv2.erode(img, kernel)
    return img


def morph_open_close(img, kernels):
    img = img.copy()
    for kernel in kernels:
        img = cv2.erode(img, kernel)
        img = cv2.dilate(img, kernel)
        img = cv2.dilate(img, kernel)
        img = cv2.erode(img, kernel)
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def subtract(img1, img2):
    return cv2.subtract(img1, img2)


def thresh(img):
    return cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]


def create_mask(img):
    return np.ones(img.shape[:2], dtype="uint8") * 255


def make_contours(img, mask):
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) <= 200:
            cv2.drawContours(mask, [contour], -1, 0, -1)


def make_bitwise(img, noise_mask):
    return cv2.bitwise_and(img, img, mask=noise_mask)


def image_processing(img):
    img_green = get_green_channel(img)

    normalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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


def get_images():
    image_list = list()
    folder_path = 'resources/pictures/'
    subfolders = ['chasedb1/', 'hrf/']
    for subfolder in subfolders:
        for file in pathlib.Path(folder_path + subfolder).iterdir():
            img = cv2.imread(str(file))
            image_list.append(img)
    return image_list


def main():
    plt.imshow(image_processing(get_images()[0]), cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
