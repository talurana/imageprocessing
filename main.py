import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
from skimage.segmentation import clear_border
from typing import Tuple, Any
from skimage import measure


def separated_colors(img: np.ndarray) -> tuple[Any, ...]:
    return tuple(cv2.split(img))


def make_histogram(img: np.ndarray, bins: int = 100, color: str = 'r', histtype: str = 'bar') -> None:
    plt.hist(img.ravel(), bins=bins, color=color, histtype=histtype)
    plt.show()


def blurring_gaussian(img: np.ndarray, k_size: Tuple[int, int] = (5, 5), sigma_x: int = 0,
                      sigma_y: int = 0) -> np.ndarray:
    return cv2.GaussianBlur(img, k_size, sigma_x, sigma_y)


def blurring_median(img: np.ndarray, k_size: int = 7) -> np.ndarray:
    return cv2.medianBlur(img, k_size)


def blurring_bilateral(img: np.ndarray, diameter: int = 7, sigmacolor: int = 95, sigmaspace: int = 95) -> np.ndarray:
    return cv2.bilateralFilter(img, diameter, sigmacolor, sigmaspace)


def binarisation(img: np.ndarray, thresh: int = 0, maxval: int = 255,
                 type_bin=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) -> int:
    _, thresh = cv2.threshold(img, thresh, maxval, type_bin)
    return thresh


def change_color_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def segmentation(img_bin: int, img: np.ndarray, kernel: np.ndarray = np.ones((3, 3), np.uint8)):
    opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = clear_border(opening)
    opening = cv2.erode(opening, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.66 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, img_markers = cv2.connectedComponents(sure_fg)
    img_markers = img_markers + 1
    img_markers[unknown == 255] = 0
    img_markers = cv2.watershed(img, img_markers)
    img[img_markers == -1] = [0, 0, 255]
    return img, img_markers


propertiesList = ['Area',
                  'equivalent_diameter',
                  'orientation',
                  'MajorAxisLength',
                  'MinorAxisLength',
                  'Perimeter',
                  'MinIntensity',
                  'MeanIntensity',
                  'MaxIntensity']

path = "img/*.bmp"
number = 0
with open('measurements_bilateral.csv', 'w') as file:
    file.write('cell #' + ',' + ',' + ','.join(propertiesList) + '\n')
    for cells in glob.glob(path):
        print(cells)
        number += 1
        image = cv2.imread(cells)
        gray = change_color_gray(image)
        gray = blurring_bilateral(gray)
        binarizated = binarisation(gray)
        final_image, markers = segmentation(binarizated, image)
        cv2.imwrite('bilateral\\' + str(number) + '_b.jpg', final_image)

        regions = measure.regionprops(markers, intensity_image=image)

        cell_number = 0
        for region_props in regions:
            if cell_number != 0:
                file.write(str(cell_number) + ',')
                for num, prop in enumerate(propertiesList):
                    file.write(',' + str(region_props[prop]))
                file.write('\n')
            cell_number += 1
