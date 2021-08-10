import cv2
import numpy as np
from PIL import Image

from utils import transform_by4


def crop_gray_image_by_contours(img_gray):
    folder = "/Users/satoakira/not-synced-picture/shogi-movie/20210809/"
    img_size = img_gray.shape[0] * img_gray.shape[1]
    print(img_size)
    # img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                         cv2.THRESH_BINARY, 11, 5)
    ret, img_binary = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(folder + "0006.jpg", img_binary)

    h, w = img_gray.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    retval, img_binary, mask, rect = cv2.floodFill(img_binary, mask, seedPoint=(0, 0), newVal=(255, 255, 255))


    imageArray = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), np.uint8)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imageArray = cv2.drawContours(imageArray, contours, -1, (0, 255, 0), 3)
    cv2.imwrite(folder + "sample.jpg", imageArray)
    edges = cv2.Canny(img_gray, 25, 50)
    cv2.imwrite(folder + "sample2.jpg", edges)
    exit(0)

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # tmpcontours = list(filter(lambda x: img_size*0.6 < cv2.contourArea(x), contours))
    # if len(tmpcontours) > 0:
    #     cv2.drawContours(img_binary, tmpcontours[0], 0, (255, 255, 255), -1)
    #     print(cv2.contourArea(tmpcontours[0]))
    #     contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite(folder + "0007.jpg", img_binary)
    # 小さい輪郭は誤検出として削除する
    contours = list(filter(lambda x: img_size * 0.6 > cv2.contourArea(x) > 100, contours))
    # cv2.drawContours(img_gray, contours, -1, color=(0, 0, 255), thickness=2)

    imageArray = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), np.uint8)

    for contour in contours:
        # x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(img_gray, (x, y), (x + w, y + h), (0, 0, 255), cv2.LINE_4)

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(imageArray, [box], 0, (255, 255, 255), -1)

    # cv2.line(imageArray, (0, 0), (0, img_gray.shape[0]), (0, 0, 0), 3)
    # cv2.line(imageArray, (0, 0), (img_gray.shape[1], 0), (0, 0, 0), 3)
    # cv2.line(imageArray, (img_gray.shape[1], 0), (img_gray.shape[1], img_gray.shape[0]), (0, 0, 0), 3)
    # cv2.line(imageArray, (0, img_gray.shape[0]), (img_gray.shape[1], img_gray.shape[0]), (0, 0, 0), 3)
    cv2.imwrite(folder + "0008.jpg", imageArray)

    ret, img_binary = cv2.threshold(cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY), 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # rect = cv2.minAreaRect(contours[0])
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(img_gray, [box], 0, (255, 255, 255), 2)
    # print("box:")
    # print(box)
    # print("rect:")
    # print(rect)
    # saveimg = transform_by4(img_gray, box)
    #
    # cv2.imwrite("/Users/satoakira/Downloads/2020-10-03_213172-0006.jpg", saveimg)
    #
    # return cv2.resize(saveimg, dsize=(int(rect[1][0]), int(rect[1][1])))

    x, y, w, h = cv2.boundingRect(contours[0])
    return img_gray[y:y+h, x:x+w]
