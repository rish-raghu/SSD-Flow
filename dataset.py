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
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
import Nvidia_SSD.src.utils as SSD_utils
#from enum import Enum

# class Views(Enum):
#    UNKNOWN = 0
#    FRONT = 1
#    FRONT_LEFT = 2
#    FRONT_RIGHT = 3
#    SIDE_LEFT = 4
#    SIDE_RIGHT = 5

# class Labels(Enum):
#   TYPE_UNKNOWN = 0;
#   TYPE_VEHICLE = 1;
#   TYPE_PEDESTRIAN = 2;
#   TYPE_SIGN = 3;
#   TYPE_CYCLIST = 4;
  


class Dataset(torch.utils.data.Dataset):
  def __init__(self, record_file, view, dboxes, augment=False, val=False, flow=False):
    # Only pass in dboxes if data augmentation is desired
    
    self.val = val

    record = tf.data.TFRecordDataset(record_file, compression_type='')
    self.frames= []
    for frame_data in record: 
      frame = open_dataset.Frame()
      frame.ParseFromString(bytearray(frame_data.numpy()))
      self.frames.append(frame)
    self.view = view
    
    self.augment = augment
    if augment:
      self.transform = SSD_utils.SSDTransformer(dboxes, val=False)
    else:
      self.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

    self.encoder = SSD_utils.Encoder(dboxes)

    if flow:
      self.flows = torch.as_tensor(np.load('flow/val/' + record_file[9:-9] + '.npy', allow_pickle=True), dtype=torch.float32)
    else:
      self.flows = None

  def __len__(self):
    return len(self.frames) if self.flows is None else len(self.frames)-1

  def get_frame(self, index):
    return self.frames[index]

  def __getitem__(self, index):
    frame = self.frames[index]

    for camera_labels in frame.camera_labels:
      if camera_labels.name != self.view: continue
      if not len(camera_labels.labels): return None
      bboxes = torch.zeros((len(camera_labels.labels), 4), dtype=torch.float32)
      labels = torch.zeros(len(camera_labels.labels), dtype=torch.int64)
      for i, label in enumerate(camera_labels.labels):
        bboxes[i, 0] = label.box.center_x - 0.5 * label.box.length
        bboxes[i, 1] = label.box.center_y - 0.5 * label.box.width
        bboxes[i, 2] = label.box.center_x + 0.5 * label.box.length
        bboxes[i, 3] = label.box.center_y + 0.5 * label.box.width
        labels[i] = label.type if label.type < 4 else label.type-1
    
    for image in frame.images:
      if image.name == self.view:
        im = torch.as_tensor(tf.image.decode_jpeg(image.image).numpy()/255, dtype=torch.float32)
        im = im.permute(2, 0, 1)
        orig_size = torch.as_tensor(im.shape[1:], dtype=torch.int16)
        bboxes[:, [0,2]] /= im.shape[2]
        bboxes[:, [1,3]] /= im.shape[1]
        
        if self.augment:  
          im = torchvision.transforms.ToPILImage()(im)
          im, _, bboxes, labels = self.transform(im, self.transform.size, bbox=bboxes, label=labels, max_num=8732)
        else:
          im = self.transform(im)
        
        bboxes, labels = self.encoder.encode(bboxes, labels)
        
        if self.flows is not None:
          return orig_size, index, im, bboxes, labels, self.flows[index]
        else:
          return orig_size, index, im, bboxes, labels

# necessary to discard samples without any objects
def collate_fn(batch):
  batch = list(filter(lambda data: data is not None, batch))
  if len(batch)>0:
    return torch.utils.data.dataloader.default_collate(batch)
  else:
    return None, None, None, None, None

if __name__ == '__main__':
  tf.enable_eager_execution()
  filename = 'data/train/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord'
  dboxes = SSD_utils.dboxes300_coco()
  dataset = Dataset(filename, open_dataset.CameraName.FRONT, dboxes)
  for i in range(190):
    if i!=150: continue
    labels = dataset.__getitem__(i)[2]
    #print(torch.sum(labels), labels.shape)
      
