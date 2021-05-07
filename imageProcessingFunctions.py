import cv2
import numpy as np


def get_green_channel(img):
    return img[:, :, 1]


def make_normalizer():
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


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
        # equivalent
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
