import os
from waymo_open_dataset import dataset_pb2
import tensorflow.compat.v1 as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
from flow import warp
import torchvision

def extract_image(frame):
  for image in frame.images:
    if image.name == dataset_pb2.CameraName.FRONT:
      image = torch.as_tensor(tf.image.decode_jpeg(image.image).numpy(), dtype=torch.float32)
      image = torchvision.transforms.Resize((300, 300))(image.permute(2, 0, 1))
      return image

def main():
  n1 = 50
  record_file = 'segment-8513241054672631743_115_960_135_960_with_camera_labels'

  record = tf.data.TFRecordDataset('data/train/'+record_file+'.tfrecord', compression_type='')
  for frame_num, frame_data in enumerate(record):
    if frame_num==n1:
      frame = dataset_pb2.Frame()
      frame.ParseFromString(bytearray(frame_data.numpy()))
      image1 = extract_image(frame)
    elif frame_num==n1+1:
      frame = dataset_pb2.Frame()
      frame.ParseFromString(bytearray(frame_data.numpy()))
      image2 = extract_image(frame)
      break

  flows = torch.as_tensor(np.load('flow/train/' + record_file + '.npy'), dtype=torch.float32)
  flow = flows[n1+1]

  plt.imshow((torch.squeeze(image1, 0).permute(1,2,0)/255))
  plt.savefig('im1')
  plt.imshow((torch.squeeze(image2, 0).permute(1,2,0)/255))
  plt.savefig('im2')

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  image1 = image1.to(device)
  flow = flow.to(device)

  warped = warp(torch.unsqueeze(image1, 0), torch.unsqueeze(flow, 0))
  plt.imshow((torch.squeeze(warped, 0).permute(1,2,0)/255).cpu())
  plt.savefig('warped12')

if __name__ == "__main__":
  main()
