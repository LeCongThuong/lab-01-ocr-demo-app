import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def resize_img(img, tgt_size=1000):
    h, w, c = img.shape
    if w > 1000:
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    return img


def read_img(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def threshold(image, max_thresh=160):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, max_thresh, 255,cv2.THRESH_BINARY_INV)
    return thresh


def dilate_img(thresh_img, kernel_size=(3, 30)):
  kernel = np.ones(kernel_size, np.uint8)
  dilated_img = cv2.dilate(thresh_img, kernel, iterations = 1)
  return dilated_img


def get_bounding_boxes(dilated_img):
    (contours, heirarchy) = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    bounding_box_lines = []
    for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)

        if h < 10 or h > 50 or w < 200:
            continue
        bounding_box_lines.append([x, y, x + w, h + y])
    return bounding_box_lines


def draw_lines(img, bounding_box_lines, line_color=(40, 100, 250), line_width=2):
  img2 = img.copy()
  for x_l, y_l, x_r, y_r in bounding_box_lines:
    cv2.rectangle(img2, (x_l,y_l), (x_r, y_r), line_color, line_width)
  return img2


def get_word_bounding_boxes(dilated_word_img, bounding_box_lines):
  bb_words_in_line_list = []
  for x_tl, y_tl, x_br, y_br in bounding_box_lines:
    roi_line = dilated_word_img[y_tl:y_br, x_tl:x_br]
    (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])

    words_list = []
    for word in sorted_contour_words:
      if cv2.contourArea(word) < 10:
          continue
      x2, y2, w2, h2 = cv2.boundingRect(word)
      if w2 < 5:
        continue
      words_list.append([x_tl+x2, y_tl+y2, x_tl+x2+w2, y_tl+y2+h2])
      bb_words_in_line_list.append(words_list)
  return bb_words_in_line_list


def draw_words(img, bb_words_in_line_list, line_color=(255,255,100), line_width=2):
  img3 = img.copy()
  for bb_word_list in bb_words_in_line_list:
    for (x_l, y_l, x_r, y_r) in bb_word_list:
      cv2.rectangle(img3, (x_l, y_l), (x_r, y_r), line_color, line_width)
  return img3


def crop_img(img, bb_word):
  word_img = img[bb_word[1]:bb_word[3], bb_word[0]:bb_word[2]]
  return word_img


def crop_all(img, bb_words_in_line_list):
  word_img_list = []
  for bb_word_list in bb_words_in_line_list:
    for bb_word in bb_word_list:
      word_img = crop_img(img, bb_word)
      word_img_list.append(word_img)
  return word_img_list


def save_img_list(word_img_list, dest_dir):
  for index, word_img in enumerate(word_img_list):
    cv2.imwrite(os.path.join(dest_dir, f"{index}.png"), word_img)


def detection(img):
    if isinstance(img, str):
        img = read_img(img)
    img = resize_img(img)
    thresh_img = threshold(img)
    dilated_img = dilate_img(thresh_img, kernel_size=(3, 30))
    bounding_box_lines = get_bounding_boxes(dilated_img)
    dilated_word_img = dilate_img(thresh_img, kernel_size=(3, 3))
    bb_words_in_line_list = get_word_bounding_boxes(dilated_word_img, bounding_box_lines)
    line_imgs = [crop_img(img, bounding_box) for bounding_box in bounding_box_lines]
    word_imgs = crop_all(img, bb_words_in_line_list)
    return bounding_box_lines, bb_words_in_line_list, line_imgs, word_imgs


