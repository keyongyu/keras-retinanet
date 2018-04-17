from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
import cv2
import scipy.ndimage
import copy


class Sku:
    def __init__(self, sku_seq, full_path):
        self.sku_seq = sku_seq
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        self.has_alpha = (img.shape[2] == 4)
        if self.has_alpha:
            alpha_channel = img[:, :, 3]
            alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
            alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
            self.bgr = img[:, :, :3].astype(np.float32) * alpha_factor
            self.alpha = alpha_factor
        else:
            self.bgr = img

    @property
    def shape(self):
        return self.bgr.shape[0:2]


def _add_sku(bgr_my_bg, x, y, sku):
    x = int(x)
    y = int(y)
    new_h = sku.alpha.shape[0]
    new_w = sku.alpha.shape[1]

    new_img = bgr_my_bg
    # bg_b, bg_g, bg_r = np.rollaxis(bgr_my_bg, -1)
    # rrr, ggg, bbb = sku.rr.copy(), sku.gg.copy(), sku.bb.copy()
    if sku.has_alpha:
        bg_img = new_img[y:y+new_h, x:x+new_w]
        sku_img = sku.bgr + bg_img*(1.0 - sku.alpha)
    else:
        sku_img = sku.bgr
    new_img[y:new_h + y, x:new_w + x] = sku_img


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        try:
            # class_name, class_id = row
            class_seq, class_code, class_name = row
            class_code = class_code + "_" + class_name
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_sequence, class_code, class_name\''.format(line)),
                       None)
        class_seq = _parse(class_seq, int, 'line {}: malformed class_sequence: {{}}'.format(line))

        if class_code in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_code] = class_seq
    return result


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class NPStudioGenerator(Generator):
    def __init__(
            self,
            data_folder,
            subset="train",
            **kwargs
    ):
        self.data_folder = data_folder
        self.sku_map = {}
        self.subset=subset
        # parse the provided class file
        # format is class_seq, class_code, class_name
        csv_class_file = os.path.join(data_folder, "label.csv")
        try:
            with _open_for_csv(csv_class_file) as file:
                # dict: key is class_code , value is class_seq
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        # dict: key is class_seq , value is class_code
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key  # class_code

        csv_data_file = os.path.join(data_folder, "index.csv")
        # csv with bg_path, sku_img_path, class_seq, x, y, x_gap, y_gap
        try:
            with _open_for_csv(csv_data_file) as file:
                self.image_instruction = self._read_instruction(csv.reader(file, delimiter=','), self.labels)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)

        super(NPStudioGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_instruction)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_bg_path(self, image_index):
        return os.path.join(self.data_folder, self.image_instruction[image_index][0])

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.image_bg_path(image_index))
        return float(image.width) / float(image.height)

    def _read_instruction(self, csv_reader, labels):
        result = []

        for line, row in enumerate(csv_reader):
            sku_seq = 0
            try:
                bg_img_file, sku_img_file, sku_seq, x, y, gap_x, gap_y = row
                sku_seq = _parse(sku_seq, int, 'line {}: malformed sku_seq: {{}}'.format(line))
                if sku_img_file not in self.sku_map:
                    self.sku_map[sku_img_file] = Sku(sku_seq, os.path.join(self.data_folder, sku_img_file))
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'bg_image_file,sku_img, sku_seq, x,y,x_gap,y_gap\' or \'img_file,,,,,\''.format(
                        line)),
                    None)

            if sku_seq not in labels:
                raise ValueError('line {}: unknown class seq: \'{}\' (labels: {})'.format(line, sku_seq, labels))

            result.append(row)
            #if line > 1000:
            #    break
        if self.subset != "train":
            return result[0:len(result):int(len(result)/1000)]
        return result

    def load_image(self, image_index):
        # return read_image_bgr(self.image_path(image_index))
        bgr_bg_img = cv2.imread(self.image_bg_path(image_index))
        # image = Image.open(self.image_bg_path(image_index))

        _, sku_img_file, sku_seq, x_, y_, col_gap, row_gap = self.image_instruction[image_index]

        sku = self.sku_map[sku_img_file]
        x_, y_ = int(x_), int(y_)
        x, y = x_, y_
        col_gap, row_gap = int(col_gap), int(row_gap)
        while True:
            # add_sku(bgr_bg_img, x, y, boxes, sku,)
            _add_sku(bgr_bg_img, x, y, sku)
            x += sku.shape[1] + col_gap
            if x + sku.shape[1] > bgr_bg_img.shape[1]:
                y += sku.shape[0] + row_gap
                x = x_

            if y + sku.shape[0] > bgr_bg_img.shape[0]:
                break
        return bgr_bg_img

    def load_annotations(self, image_index):
        image = Image.open(self.image_bg_path(image_index))

        _, sku_img_file, sku_seq, x_, y_, col_gap, row_gap = self.image_instruction[image_index]
        sku = self.sku_map[sku_img_file]
        boxes = []
        x_, y_ = int(x_), int(y_)
        x, y = x_, y_
        col_gap, row_gap = int(col_gap), int(row_gap)
        while True:
            boxes.append({"x1": x, "y1": y, "x2": x + sku.shape[1], "y2": y + sku.shape[0]})
            x += sku.shape[1] + col_gap
            if x + sku.shape[1] > image.width:
                y += sku.shape[0] + row_gap
                x = x_

            if y + sku.shape[0] > image.height:
                break

        np_boxes = np.zeros((len(boxes), 5))
        for idx, box in enumerate(boxes):
            np_boxes[idx, 0] = float(box['x1'])
            np_boxes[idx, 1] = float(box['y1'])
            np_boxes[idx, 2] = float(box['x2'])
            np_boxes[idx, 3] = float(box['y2'])
            np_boxes[idx, 4] = sku.sku_seq

        return np_boxes
