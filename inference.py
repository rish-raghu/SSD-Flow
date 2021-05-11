import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from time import time
from dataset import Dataset, collate_fn
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from models import SSD_Baseline, SSD_Flow, SSD_Ablation
import Nvidia_SSD.src.utils as SSD_utils

metrics_type = 'lib'

def inference(model_path, eval_path, pred_save_path, view=dataset_pb2.CameraName.FRONT):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  with torch.no_grad():
    model = SSD_Ablation()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    model = model.to(device)

    dboxes = SSD_utils.dboxes300_coco()
    decoder = SSD_utils.Encoder(dboxes)
    scene_files = sorted(os.listdir(eval_path))

    pred_objects = metrics_pb2.Objects()

    for scene_num, record_file in enumerate(scene_files):
      
      scene_data = Dataset(os.path.join(eval_path, record_file), view, dboxes, val=True, flow=False)
      scene_dataloader = torch.utils.data.DataLoader(scene_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

      for batch_num, (orig_size, idxs, images, bboxes, labels) in enumerate(scene_dataloader):
        if images is None: 
          print("None")
          continue
        images = images.to(device)
        bboxes = bboxes.to(device)
        labels = labels.to(device)
        #flows = flows.to(device)
        
        pred_boxes, pred_scores = model(images)
        preds = decoder.decode_batch(pred_boxes, pred_scores, criteria=0.5, max_output=100)

        if metrics_type=='lib':
          for i, pred in enumerate(preds):
            create_prediction_file(pred, orig_size[i], record_file, idxs[i], pred_save_path)
        elif metrics_type=='native':
          for i, pred in enumerate(preds):
            frame = scene_data.get_frame(idxs[i])
            for j in range(len(pred[0])):
              pred_objects.objects.append(create_prediction_obj(frame, pred[0][j], pred[2][j], pred[1][j], view, orig_size[i]))
          f = open(pred_save_path, 'wb')
          f.write(pred_objects.SerializeToString())
          f.close()
        
      print("Finished scene", scene_num)

# For library metrics
def create_prediction_file(pred, orig_size, record_file, idx, pred_save_path):
  label_map = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cycle'}
  boxes, labels, scores = pred
  with open(pred_save_path + '/' + record_file + '_' + str(idx.item()) + '.txt', 'w') as f:
    for i in range(len(labels)):
      f.write(str(label_map[labels[i].item()])); f.write(' ')
      f.write(str(scores[i].item())); f.write(' ')
      f.write(str((boxes[i][0] * orig_size[1]).item())); f.write(' ')
      f.write(str((boxes[i][1] * orig_size[0]).item())); f.write(' ')
      f.write(str((boxes[i][2] * orig_size[1]).item())); f.write(' ')
      f.write(str((boxes[i][3] * orig_size[0]).item())); f.write('\n')
    f.close()

# For native metrics; Based on example in from waymo-open-dataset/waymo_open_dataset/metrics/tools/create_prediction_file_example.py
def create_prediction_obj(frame, pred_box, pred_score, pred_label, view, orig_size):
  
  o = metrics_pb2.Object()
  o.context_name = frame.context.name
  o.frame_timestamp_micros = frame.timestamp_micros
  o.camera_name = view

  box = label_pb2.Label.Box()
  box.center_x = (pred_box[0] + pred_box[2])/2 * orig_size[1]
  box.center_y = (pred_box[1] + pred_box[3])/2 * orig_size[0]
  box.length = (pred_box[2] - pred_box[0]) * orig_size[1]
  box.width = (pred_box[3] - pred_box[1]) * orig_size[0]
  
  o.object.box.CopyFrom(box)
  
  # This must be within [0.0, 1.0]. It is better to filter those boxes with
  # small scores to speed up metrics computation.
  o.score = pred_score
  # Use correct type.
  o.object.type = pred_label+1 if pred_label==3 else pred_label

  return o

  # Add more objects. Note that a reasonable detector should limit its maximum
  # number of boxes predicted per frame. A reasonable value is around 400. A
  # huge number of boxes can slow down metrics computation.

if __name__ == "__main__":
  inference(sys.argv[1], sys.argv[2], sys.argv[3])

