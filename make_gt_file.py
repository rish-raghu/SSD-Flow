import os
import sys
import numpy as np
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
import tensorflow.compat.v1 as tf

metric_type = 'lib'

def make_gt_file_lib(data_path, gt_save_path, view=dataset_pb2.CameraName.FRONT):
  scene_files = sorted(os.listdir(data_path))
  label_map = {1: 'Vehicle', 2: 'Pedestrian', 4: 'Cycle'}
  # areas = []
  for scene_num, record_file in enumerate(scene_files):
    record = tf.data.TFRecordDataset(os.path.join(data_path, record_file), compression_type='')
    
    for frame_num, frame_data in enumerate(record): 
      frame = dataset_pb2.Frame()
      frame.ParseFromString(bytearray(frame_data.numpy()))
      
      for camera_labels in frame.camera_labels:
        if camera_labels.name != view: continue
        with open(gt_save_path + '/' + record_file + '_' + str(frame_num) + '.txt', 'w') as f:
          for label in camera_labels.labels:
            f.write(label_map[label.type]); f.write(' ')
            f.write(str(label.box.center_x - 0.5 * label.box.length)); f.write(' ')
            f.write(str(label.box.center_y - 0.5 * label.box.width)); f.write(' ')
            f.write(str(label.box.center_x + 0.5 * label.box.length)); f.write(' ')
            f.write(str(label.box.center_y + 0.5 * label.box.width)); f.write('\n')
          f.close()
             
    print("Finished scene", scene_num)
  # print(np.mean(areas))
  # print(np.median(areas))


def make_gt_file_native(data_path, gt_save_path, view=dataset_pb2.CameraName.FRONT):
  
  scene_files = sorted(os.listdir(data_path))
  gt_objects = metrics_pb2.Objects()

  for scene_num, record_file in enumerate(scene_files):
    record = tf.data.TFRecordDataset(os.path.join(data_path, record_file), compression_type='')
    
    for frame_data in record: 
      frame = dataset_pb2.Frame()
      frame.ParseFromString(bytearray(frame_data.numpy()))
      
      for camera_labels in frame.camera_labels:
        if camera_labels.name != view: continue
        for label in camera_labels.labels:
          gt_objects.objects.append(create_gt_obj(frame, label, view))
    
    print("Finished scene", scene_num)
    
  f = open(gt_save_path, 'wb')
  f.write(gt_objects.SerializeToString())
  f.close()


def create_gt_obj(frame, label, view):
  
  o = metrics_pb2.Object()
  o.context_name = frame.context.name
  o.frame_timestamp_micros = frame.timestamp_micros
  o.camera_name = view

  box = label_pb2.Label.Box()
  box.center_x, box.center_y = label.box.center_x, label.box.center_y
  box.length, box.width = label.box.length, label.box.width
  o.object.box.CopyFrom(label.box)
  
  o.object.type = label.type

  return o

if __name__ == "__main__":
  if metric_type=='lib':
    make_gt_file_lib(sys.argv[1], sys.argv[2])
  elif metric_type=='native':
    make_gt_file_native(sys.argv[1], sys.argv[2])