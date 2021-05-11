import torch
import numpy as np 
import matplotlib.pyplot as plt
import sys
from models import SSD_Baseline, SSD_Flow
import Nvidia_SSD.src.utils as SSD_utils
from dataset import Dataset
from waymo_open_dataset import dataset_pb2
import os

scene_folder = sys.argv[1]
scene_num = int(sys.argv[2])
frame_num = int(sys.argv[3])
model_path = sys.argv[4]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SSD_Flow()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict['model_state_dict'])
model.eval()
model = model.to(device)

dboxes = SSD_utils.dboxes300_coco()
decoder = SSD_utils.Encoder(dboxes)
scene_file = sorted(os.listdir(scene_folder))[scene_num]
dataset = Dataset(os.path.join(scene_folder, scene_file), dataset_pb2.CameraName.FRONT, dboxes, flow=True)

orig_size, index, im, bboxes, labels, flow = dataset[frame_num]
with torch.no_grad():
  im = im.to(device)
  flow = flow.to(device)
  pred_boxes, pred_scores = model(torch.unsqueeze(im, 0), torch.unsqueeze(flow, 0))
  preds = decoder.decode_batch(pred_boxes, pred_scores, criteria=0.5, max_output=5)

im = im+1
im = im - im.min()
im = im / (im.max() - im.min())
SSD_utils.draw_patches(im.cpu().permute(1,2,0), preds[0][0].cpu(), preds[0][1].cpu(), order='ltrb', label_map={1:'car', 2:'person', 3:'cyclist'})