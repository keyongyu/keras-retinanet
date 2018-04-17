"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import cv2
import pytest
from keras_retinanet.preprocessing import npstudio_generator

from keras_retinanet.utils.visualization import draw_annotations, draw_boxes

from keras_retinanet.utils.transform import random_transform_generator

def test_npstudio_generator():
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.0,
        flip_y_chance=0.0,
    )
    generator = npstudio_generator.NPStudioGenerator(
        "/home/keyong/Downloads/studio_watson/studio_data",
        subset="val",
        transform_generator=transform_generator)
    print("count=" + str(generator.size()))
    for i in range(generator.size()):
        image = generator.load_image(i)
        annotations = generator.load_annotations(i)
        draw_annotations(image, annotations, color=(0, 0, 255), generator=generator)
        #cv2.imshow('Image', image)
        cv2.imwrite("/home/keyong/test/%5d.jpg" %(i),image)
        #break



