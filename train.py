import torch
import numpy as np 
import pandas as pd
import sys
import os
from time import time
from dataset import Dataset, collate_fn
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from models import SSD_Baseline, SSD_Flow, SSD_Ablation
from Nvidia_SSD.src.model import Loss
import Nvidia_SSD.src.utils as SSD_utils


def train(train_path, model_save_path, num_epochs=3, model_load_path=None, patience=2, gamma=0.1, imp_thresh=0.001):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  if model_load_path:
    model = SSD_Ablation()
    state_dict = torch.load(model_load_path)
    model.load_state_dict(state_dict['model_state_dict'])
  else:
    model = SSD_Ablation(initialize=True)
  model.freeze(freeze_add=False)
  model.train_setbn(bn_eval=False) # need to set BN to eval mode if using batch size of 1
  model = model.to(device)

  trainable_params = [param for param in model.parameters() if param.requires_grad]
  optimizer = torch.optim.SGD(trainable_params, lr=0.001, momentum=0.9, weight_decay=0.0005)
  if model_load_path: optimizer.load_state_dict(state_dict['optimizer_state_dict'])
  
  start_epoch = state_dict['epoch']+1 if model_load_path else 0
  # prev_best_loss = state_dict['loss'] if model_load_path else float('inf')
  # num_epochs_no_imp = 0
  # imp_this_lr = False

  dboxes = SSD_utils.dboxes300_coco()
  loss_criterion = Loss(dboxes)

  scene_files = sorted(os.listdir(train_path))

  for epoch in range(start_epoch, num_epochs):
    
    np.random.shuffle(scene_files)
    epoch_loss = 0

    for scene_num, record_file in enumerate(scene_files):
      
      #try:
      scene_data = Dataset(os.path.join(train_path, record_file), dataset_pb2.CameraName.FRONT, dboxes, augment=False)
      #except:
        #print(record_file, "Corrupted!")
        #continue
      scene_dataloader = torch.utils.data.DataLoader(scene_data, batch_size=32, shuffle=True, collate_fn=collate_fn, drop_last=True)
      scene_loss = 0
      
      start = time()
      for frame_num, (_, _, images, bboxes, labels) in enumerate(scene_dataloader):
        if len(labels) == 1: continue # batch size of 1 will cause error

        images = images.to(device)
        bboxes = bboxes.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        pred_boxes, pred_scores = model(images)
        loss = loss_criterion(pred_boxes, pred_scores, bboxes.transpose(1,2), labels)
        loss.backward()
        optimizer.step()
        scene_loss += loss.item()
      
      scene_loss /= len(scene_data)
      epoch_loss += scene_loss
      print("Scene", scene_num, "loss:", scene_loss)
      if (scene_num+1)%50==0:
        torch.save({'epoch': epoch, 'scene_num': scene_num, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': scene_loss},
                os.path.join(model_save_path, "ep_"+str(epoch)+'_sc_'+str(scene_num)))

    epoch_loss /= len(scene_files)
    print("\nEpoch", epoch, "loss:", epoch_loss, "\n")
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': epoch_loss},
                os.path.join(model_save_path, "ep_"+str(epoch)))
    
    # LR stepping and early stopping
    # if epoch_loss > prev_best_loss - imp_thresh:
    #   num_epochs_no_imp +=1
    #   if num_epochs_no_imp == patience:
    #     if imp_this_lr:
    #       for g in optimizer.param_groups:
    #         g['lr'] *= gamma
    #       imp_this_lr = False
    #     else:
    #       return
    # else:
    #   num_epochs_no_imp = 0
    #   imp_this_lr = True
    #   prev_best_loss = epoch_loss
        

if __name__ == "__main__":
  if len(sys.argv) > 1:
    train("data/train_full", "models/ablation", model_load_path=sys.argv[1])
  else:
    train("data/train_full", "models/ablation")