import cv2
import numpy as np

MIN_AREA_RATIO = 0.04
MIN_CIRCLENESS = 0.6

def filter_contour(contour, total_area):
    area = cv2.contourArea(contour)
    rA = np.sqrt(area / np.pi)
    rC = cv2.arcLength(contour, True) / (2 * np.pi)
    circleness = rA / rC
    return (area / total_area) > MIN_AREA_RATIO and circleness > MIN_CIRCLENESS

def crop(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (img[y:y + h, x:x + w].copy(), x, y)

def detect_cards(img):
    _, _, img_value = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    total_area = img_value.shape[0] * img_value.shape[1]

    ret, img_thresh = cv2.threshold(img_value, 100, 255, cv2.THRESH_OTSU)
    kernel = np.ones((5, 5))
    opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [x for x in contours if filter_contour(x, total_area)]

    return [crop(img, contour) for contour in filtered_contours]