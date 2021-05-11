import os
import sys
from waymo_open_dataset import dataset_pb2
import tensorflow.compat.v1 as tf
import torch
import torchvision
from RAFT_flow.core.raft import RAFT
from RAFT_flow.core.utils.utils import InputPadder
import cv2
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def compute_flows(data_path, save_path):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  args = dict_obj({'mixed_precision': True})
  model = torch.nn.DataParallel(RAFT(args))
  model.load_state_dict(torch.load('code/RAFT_flow/raft-kitti.pth'))
  model.eval()
  model = model.to(device)

  scene_files = sorted(os.listdir(data_path))
  for scene_num, record_file in enumerate(scene_files[39:68]):
    try:
      dataset = Flow_Dataset(os.path.join(data_path, record_file))
    except:
      print("Corrupted!")
      print(record_file)
      continue
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    flows = np.zeros([len(dataset), 2, 300, 300])

    for idx, curr_image in dataloader:
      curr_image = curr_image.to(device)
   
      if idx > 0:
        padder = InputPadder(prev_image.shape, mode='kitti')
        prev_image_pad, curr_image_pad = padder.pad(prev_image, curr_image)
        _, flow = model(curr_image_pad, prev_image_pad, test_mode=True)
        flow = padder.unpad(flow[0])
        flows[idx,...] = flow.cpu().numpy()
      prev_image = curr_image
    
    np.save(os.path.join(save_path, record_file[:-9] + '.npy'), flows)    
    print("Finished scene", scene_num)


class Flow_Dataset(torch.utils.data.Dataset):
  def __init__(self, record_file):
    record = tf.data.TFRecordDataset(record_file, compression_type='')
    self.frames= []
    for frame_data in record: 
      frame = dataset_pb2.Frame()
      frame.ParseFromString(bytearray(frame_data.numpy()))
      self.frames.append(frame)

  def __len__(self):
    return len(self.frames)

  def __getitem__(self, idx):
    frame = self.frames[idx]
    for image in frame.images:
      if image.name != dataset_pb2.CameraName.FRONT: continue
      image = torch.as_tensor(tf.image.decode_jpeg(image.image).numpy(), dtype=torch.float32)
      image = torchvision.transforms.Resize((300, 300))(image.permute(2, 0, 1))
      return idx, image

# https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class dict_obj(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

# Source: https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
def warp(x, flo):
  """
  warp an image/tensor (im2) back to im1, according to the optical flow
  x: [B, C, H, W] (im2)
  flo: [B, 2, H, W] flow
  """
  #print(x.shape, flo.shape)
  B, C, H, W = x.size()
  # mesh grid 
  xx = torch.arange(0, W).view(1,-1).repeat(H,1)
  yy = torch.arange(0, H).view(-1,1).repeat(1,W)
  xx = xx.view(1,1,H,W).repeat(B,1,1,1)
  yy = yy.view(1,1,H,W).repeat(B,1,1,1)
  grid = torch.cat((xx,yy),1).float()

  if x.is_cuda:
      grid = grid.cuda()
  vgrid = torch.autograd.Variable(grid) + flo

  # scale grid to [-1,1] 
  vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
  vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

  vgrid = vgrid.permute(0,2,3,1)        
  output = torch.nn.functional.grid_sample(x, vgrid)
  mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
  mask = torch.nn.functional.grid_sample(mask, vgrid)

  # if W==128:
      # np.save('mask.npy', mask.cpu().data.numpy())
      # np.save('warp.npy', output.cpu().data.numpy())
  
  mask[mask<0.9999] = 0
  mask[mask>0] = 1
  
  return output*mask


if __name__ == "__main__":
  compute_flows(sys.argv[1], sys.argv[2])
