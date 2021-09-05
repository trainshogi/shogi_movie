# --------------------------------------------------------------
""" get image, points and confirm the place of piece.

Args:
  target_img_path:
    confirm image path of recognition
  base_img_path:
    base image path which is set pieces as default
  points:
    points of 4 board corners

Returns:
  A dict which is placed on the board.

Raises:
  not set
"""
# --------------------------------------------------------------

import cv2
import numpy as np
from PIL import Image
from utils import pil2cv, cv2pil, transform_by4, remove_lines, scale_to_height
from contours import crop_gray_image_by_contours

# static variable
threshold = 0.65
rotates = [20, 15, 10, 8, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -8,  -10, -15, -20]
initial_places = [(0, 6), (1, 8), (2, 8), (5, 8), (4, 8), (1, 7), (7, 7)]
initial_places = [(0, 2), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (7, 1)]
initial_places = [(0, 8)] # (0, 6), , (2, 8), (5, 8), (4, 8), (1, 7), (7, 7)
komas = ["koma/fu.png", "koma/kyo.png", "koma/kei.png", "koma/kin.png", "koma/gin.png", "koma/ka.png", "koma/hi.png", "koma/ou.png",
         "koma/vfu.png", "koma/vkyo.png", "koma/vkei.png", "koma/vkin.png", "koma/vgin.png", "koma/vka.png", "koma/vou.png"]
komas = [("koma/fu.png", 9), ("koma/kyo.png", 2), ("koma/kei.png", 2), ("koma/kin.png", 2), ("koma/gin.png", 2), ("koma/ka.png", 1), ("koma/hi.png", 1), ("koma/ou.png", 1),
         ("koma/vfu.png", 9), ("koma/vkyo.png", 2), ("koma/vkei.png", 2), ("koma/vkin.png", 2), ("koma/vgin.png", 2), ("koma/vka.png", 1), ("koma/vhi.png", 1), ("koma/vou.png", 1)] #

komas = [("koma/fu.jpg",  9, [(18, 23), (3, 187), (189, 189), (141, 22), (79, 4)], 46),
         ("koma/kyo.jpg", 2, [(37, 18), (3, 192), (189, 200), (160, 26), (98, 3)], 46),
         ("koma/kei.jpg", 2, [(57, 19), (3, 190), (196, 200), (194, 29), (132, 4)], 48),
         ("koma/gin.jpg", 2, [(67, 21), (3, 206), (210, 213), (206, 31), (139, 6)], 56),
         ("koma/kin.jpg", 2, [(23, 29), (5, 239), (237, 240), (182, 27), (96, 2)], 52),
         ("koma/ka.jpg",  1, [(44, 24), (4, 245), (243, 252), (207, 31), (128, 3)], 60),
         ("koma/hi.jpg",  1, [(53, 28), (2, 247), (237, 252), (212, 31), (135, 4)], 62),
         ("koma/ou.jpg",  1, [(73, 30), (5, 250), (247, 256), (237, 32), (157, 5)], 60),
         ("koma/to.jpg", 2, [(40, 21), (5, 184), (185, 194), (157, 25), (98, 4)], 46),
         ("koma/an.jpg", 2, [(39, 21), (6, 195), (188, 199), (160, 26), (98, 4)], 50),
         ("koma/nkei.jpg", 2, [(51, 23), (9, 194), (204, 198), (184, 27), (116, 5)], 50),
         ("koma/zen.jpg",  1, [(58, 28), (6, 208), (211, 213), (196, 31), (128, 5)], 52),
         ("koma/uma.jpg",  1, [(45, 31), (8, 249), (247, 255), (206, 33), (123, 5)], 56),
         ("koma/ryu.jpg",  1, [(57, 29), (9, 247), (240, 251), (214, 32), (134, 5)], 58)]

# komas = [("koma/kyo.jpg", 2, [(37, 18), (3, 192), (189, 200), (160, 26), (98, 3)], 50)]

# variable
masu_size = 64
ban_size = int(masu_size*9)
folder = "/Users/satoakira/not-synced-picture/shogi-movie/20210812/"
target_img_path = folder + "20210812-0009.jpg"
base_img_path = folder + "20210812-0001.jpg"
# points = [(259, 104), (267, 694), (779, 686), (794, 115)]
# points = [(459, 98), (452, 875), (1182, 866), (1157, 90)]
points = [(36, 23), (33, 1065), (969, 1049), (972, 50)]
locs = []
transformed = transform_by4(cv2.imread(target_img_path), points)
print(transformed.shape)
ratio = transformed.shape[0]/transformed.shape[1]
saveimg = cv2.resize(transformed, dsize=(ban_size, int(ban_size*ratio)))
print(saveimg.shape)
gray = cv2.cvtColor(saveimg, cv2.COLOR_RGB2GRAY)
cv2.imwrite(folder + "0007.jpg", gray)

# debug variable
img_rgb = cv2.imread(folder + "0001.jpg")

# create cropped board image
cropped_base_img = transform_by4(cv2.imread(base_img_path), points)
cropped_h, cropped_w = cropped_base_img.shape[:2]

cv2.imwrite(folder + "0004.jpg", cropped_base_img)

# removed_lines = remove_lines(cropped_base_img)
# cv2.imwrite("/Users/satoakira/Downloads/2020-10-03_213172-0008.jpg", removed_lines)

cropped_base_img_gray = cv2.equalizeHist(cv2.cvtColor(cropped_base_img, cv2.COLOR_RGB2GRAY))
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cropped_base_img_gray = clahe.apply(cv2.cvtColor(cropped_base_img, cv2.COLOR_RGB2GRAY))
# cropped_base_img_gray = cv2.cvtColor(cropped_base_img, cv2.COLOR_RGB2GRAY)

# for initial_place in initial_places:
#     y1 = int(initial_place[1]*cropped_h/9)
#     y2 = int((initial_place[1]+1)*cropped_h/9)
#     x1 = int(initial_place[0]*cropped_w/9)
#     x2 = int((initial_place[0]+1)*cropped_w/9)
#
#     one_square = cropped_base_img[y1:y2, x1:x2]
#     one_square_gray = cropped_base_img_gray[y1:y2, x1:x2]
#     cv2.imwrite("/Users/satoakira/Downloads/2021-08-08_1020-0005.jpg", one_square_gray)
#     # x, y, w, h = crop_gray_image_by_contours(cv2.cvtColor(one_square, cv2.COLOR_RGB2GRAY))
#     one_square = crop_gray_image_by_contours(one_square_gray)
#     exit(0)
#     # cv2.imwrite("/Users/satoakira/Downloads/2020-10-03_213172-0005.jpg", one_square)
#     original_tmp = cv2pil(one_square)
#     # h, w = original_tmp.size
#     # original_tmp = original_tmp.crop((w * 0.3, h * 0.2, w * 0.7, h * 0.8))
#     # original_tmp = original_tmp.crop((x, y, x + w, y + h))
#     original_tmp.save(folder + "0005.jpg")

for koma in komas:
    for omoteura in [True, False]:
        # as debug
        print(folder + koma[0])
        # original_tmp_cv2 = cv2.equalizeHist(cv2.cvtColor(cv2.imread(folder + koma[0]), cv2.COLOR_RGB2GRAY))
        original_tmp_cv2 = cv2.cvtColor(cv2.imread(folder + koma[0]), cv2.COLOR_RGB2GRAY)
        h, w = original_tmp_cv2.shape[:2]
        # original_tmp_cv2 = cv2.rectangle(original_tmp_cv2, (0, 0), (w, h), (255, 255, 255), 10)
        mask_cv2 = cv2.fillConvexPoly(np.zeros((h, w), dtype=np.uint8), np.array(koma[2]), (255, 255, 255))
        for koma_size in [64, 62, 60, 58, 56, 54, 52, 50, 48, 47, 46, 45, 43, 44, 42, 40, 37, 35]:
            if koma_size != koma[3]:
                continue
            mask = scale_to_height(cv2pil(mask_cv2), koma_size)
            original_tmp = scale_to_height(cv2pil(original_tmp_cv2), koma_size)
            if omoteura:
                mask = mask.rotate(180)
                original_tmp = original_tmp.rotate(180)
            original_tmp.save(folder + "0004.jpg")
            mask.save(folder + "0005.jpg")
            rotated_gray_tmps = []

            for rotate in rotates:
                rotated_tmp = original_tmp.rotate(angle=rotate, resample=Image.BICUBIC, expand=False) #, fillcolor=(255, 255, 255, 0))
                rotated_mask = mask.rotate(angle=rotate, resample=Image.BICUBIC, expand=False) #, fillcolor=(255, 255, 255, 0))
                h, w = rotated_tmp.size
                # print(rotated_tmp.size)
                cropped_tmp = rotated_tmp.crop((h * 0, w * 0, h * 1, w * 1))
                cropped_mask = rotated_mask.crop((h * 0, w * 0, h * 1, w * 1))
                # th, img = cv2.threshold(cv2.cvtColor(pil2cv(cropped_tmp), cv2.COLOR_RGB2GRAY), 128, 192, cv2.THRESH_OTSU)
                # th, img = cv2.threshold(pil2cv(cropped_tmp), 127, 255, cv2.THRESH_OTSU)
                # th, img = cv2.threshold(pil2cv(cropped_tmp), 127, 255, cv2.THRESH_BINARY)
                # img = cv2.adaptiveThreshold(cv2.bitwise_not(pil2cv(cropped_tmp)), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
                th, img = cv2.threshold(cv2.bitwise_not(pil2cv(cropped_tmp)), 127, 255, cv2.THRESH_BINARY)
                img = cv2.bitwise_not(img)
                img = pil2cv(cropped_tmp)
                cv2.imwrite(folder + "0001.jpg", img)

                img_mask = pil2cv(cropped_mask)
                cv2.imwrite(folder + "0006.jpg", img_mask)

                # h, w = img.shape[:2]
                # mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                # retval, img, mask, rect = cv2.floodFill(img, mask, seedPoint=(0, 0), newVal=(255, 255, 255))
                # retval, img, mask, rect = cv2.floodFill(img, mask, seedPoint=(w-1, 0), newVal=(255, 255, 255))
                # retval, img, mask, rect = cv2.floodFill(img, mask, seedPoint=(0, h-1), newVal=(255, 255, 255))
                # retval, img, mask, rect = cv2.floodFill(img, mask, seedPoint=(w-1, h-1), newVal=(255, 255, 255))

                rotated_gray_tmps.append((img, img_mask))

            locs = []
            for (temp, cropped_mask) in rotated_gray_tmps:
                cv2.imwrite(folder + "0002.jpg", temp)
                h, w = temp.shape
                match = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED, mask=cropped_mask)
                loc = np.where(match >= threshold)
                # print(match)
                # print(loc)
                locs.append(loc)

            found_places = []
            found_locs = []

            for loc in locs:
                for pt in zip(*loc[::-1]):
                    found_place = (int((pt[0] + w/2)/masu_size), int((pt[1] + h/2)/(masu_size*ratio)))
                    found_places.append(found_place)
                    found_locs.append(pt)
                    # cv2.rectangle(saveimg, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

            print(koma, koma_size, set(found_places))
            # if len(set(found_places)) == koma[1]:
            print(koma, koma_size, len(set(found_places)), set(found_places))
            # print("found_place.size = " + str(len(found_places)))
            # print("found_place.unique.size = " + str(len(set(found_places))))
            # print("found_place.unique.size is enough.")
            for pt in found_locs:
                cv2.rectangle(saveimg, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            break

cv2.imwrite(folder + "0003.jpg", saveimg)
