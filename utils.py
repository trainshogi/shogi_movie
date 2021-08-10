import cv2
import numpy as np
from PIL import Image


def pil2cv(image):
    """ PIL型 -> OpenCV型 """
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image):
    """ OpenCV型 -> PIL型 """
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


# 縦指定でPIL画像のリサイズ
def scale_to_height(img, height):
    width = round(img.width * height / img.height)
    return img.resize((width, height))


# 4点を指定してトリミングする
def transform_by4(img, points):
    points = sorted(points, key=lambda x: x[1])  # yが小さいもの順に並び替え。
    top = sorted(points[:2], key=lambda x: x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
    bottom = sorted(points[2:], key=lambda x: x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
    points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。
    width = max(np.sqrt(((points[0][0] - points[2][0]) ** 2) * 2),
                np.sqrt(((points[1][0] - points[3][0]) ** 2) * 2))
    height = max(np.sqrt(((points[0][1] - points[2][1]) ** 2) * 2),
                 np.sqrt(((points[1][1] - points[3][1]) ** 2) * 2))
    dst = np.array([
        np.array([0, 0]),
        np.array([width - 1, 0]),
        np.array([width - 1, height - 1]),
        np.array([0, height - 1]),
    ], np.float32)
    trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
    return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く


def remove_lines(img):
    h, w = img.shape[:2]
    h_space = h / 9
    w_space = w / 9
    for i in range(9):
        x1 = int(i * w_space)
        x2 = int(i * w_space)
        y1 = 0
        y2 = h
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    for i in range(9):
        x1 = 0
        x2 = w
        y1 = int(i * h_space)
        y2 = int(i * h_space)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    return img
