import torch
import torchvision
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
import tensorflow.compat.v1 as tf
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
import Nvidia_SSD.src.utils as SSD_utils

scene_folder = sys.argv[1]

num_labels = np.zeros(5)
for i, scene_file in enumerate(os.listdir(scene_folder)):
  record = tf.data.TFRecordDataset(os.path.join(scene_folder, scene_file), compression_type='')
  for frame_data in record: 
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(frame_data.numpy()))

    for camera_labels in frame.camera_labels:
      if camera_labels.name != dataset_pb2.CameraName.FRONT: continue
      for j, label in enumerate(camera_labels.labels):
        # bboxes[i, 0] = label.box.center_x - 0.5 * label.box.length
        # bboxes[i, 1] = label.box.center_y - 0.5 * label.box.width
        # bboxes[i, 2] = label.box.center_x + 0.5 * label.box.length
        # bboxes[i, 3] = label.box.center_y + 0.5 * label.box.width
        num_labels[label.type] += 1
  print("Finished scene", i)

print(num_labels)