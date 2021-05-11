import torch
from Nvidia_SSD.src.model import SSD300
from flow import warp
import sys

class SSD_Baseline(SSD300):
  # note: inherits from Nvidia SSD implementation, but self.num_classes must be
  # changed in that code 

  def __init__(self, initialize=False):
    super().__init__()
    if initialize:
      state_dict = torch.load('data/weights/nvidia_ssdpyt_fp16_190826.pt')
      for layer_name in list(state_dict['model'].keys()):
        if 'conf' in layer_name or 'loc.' in layer_name:
          del state_dict['model'][layer_name]
      self.load_state_dict(state_dict, strict=False)

  def train_setbn(self, bn_eval=True):
    self.train()
    if bn_eval: self._eval_helper(next(self.modules()))

  def _eval_helper(self, module):
    if isinstance(module, torch.nn.BatchNorm2d): 
      module.eval()
    for child in module.children():
      self._eval_helper(child)

  def freeze(self, freeze_add=True):
    for name, param in self.named_parameters():
      if 'conf' in name or 'loc.' in name:
        param.requires_grad = True
      elif not freeze_add and 'additional' in name:
        param.requires_grad = True
      else:
        param.requires_grad = False

  def unfreeze(self):
    for name, param in self.named_parameters():
      if 'bn' not in name:
        param.requires_grad = True


class SSD_Flow(SSD300):
  def __init__(self, initialize=False):
    super().__init__()
    
    self.merge_layers = []
    for num_channels in self.feature_extractor.out_channels:
      layer = torch.nn.Sequential(
                  torch.nn.Conv2d(2*num_channels, num_channels, kernel_size=3, padding=1),
                  torch.nn.BatchNorm2d(num_channels),
                  torch.nn.ReLU(inplace=True),
                  )
      self.merge_layers.append(layer)
    self.merge_layers = torch.nn.ModuleList(self.merge_layers)

    if initialize:
      state_dict = torch.load('data/weights/nvidia_ssdpyt_fp16_190826.pt')
      for layer_name in list(state_dict['model'].keys()):
        if 'conf' in layer_name or 'loc.' in layer_name:
          del state_dict['model'][layer_name]
      self.load_state_dict(state_dict, strict=False)


  def forward(self, images, flows):    
    
    curr_x = self.feature_extractor(images)
    prev_x = torch.roll(curr_x, 1, 0)
    prev_x[0,...] = 0
    flows = torch.nn.functional.interpolate(flows, size=(prev_x.shape[2], prev_x.shape[3]))
    prev_x = warp(prev_x, flows)
    x = torch.cat((curr_x, prev_x), dim=1)
    x = self.merge_layers[0](x)

    detection_feed = [x]
    for l, m in zip(self.additional_blocks, self.merge_layers[1:]):
      curr_x = l(curr_x)
      prev_x = torch.roll(curr_x, 1, 0)
      prev_x[0,...] = 0
      flows = torch.nn.functional.interpolate(flows, size=(prev_x.shape[2], prev_x.shape[3]))
      prev_x = warp(prev_x, flows)
      x = torch.cat((curr_x, prev_x), dim=1)
      x = m(x)
      detection_feed.append(x)
    
    # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
    locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

    # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
    return locs, confs

  def freeze(self, freeze_add=True):
    for name, param in self.named_parameters():
      if 'conf' in name or 'loc.' in name or 'merge' in name:
        param.requires_grad = True
      elif not freeze_add and 'additional' in name:
        param.requires_grad = True
      else:
        param.requires_grad = False

  def train_setbn(self, bn_eval=True):
    self.train()
    if bn_eval: self._eval_helper(next(self.modules()))

  def _eval_helper(self, module):
    if isinstance(module, torch.nn.BatchNorm2d): 
      module.eval()
    for child in module.children():
      self._eval_helper(child)


class SSD_Ablation(SSD300):
  def __init__(self, initialize=False):
    super().__init__()
    
    self.extra_conv = []
    for num_channels in self.feature_extractor.out_channels:
      layer = torch.nn.Sequential(
                  torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                  torch.nn.BatchNorm2d(num_channels),
                  torch.nn.ReLU(inplace=True),
                  )
      self.extra_conv.append(layer)
    self.extra_conv = torch.nn.ModuleList(self.extra_conv)

    if initialize:
      state_dict = torch.load('data/weights/nvidia_ssdpyt_fp16_190826.pt')
      for layer_name in list(state_dict['model'].keys()):
        if 'conf' in layer_name or 'loc.' in layer_name:
          del state_dict['model'][layer_name]
      self.load_state_dict(state_dict, strict=False)


  def forward(self, images):    
    
    x = self.feature_extractor(images)
    x_conv = self.extra_conv[0](x)

    detection_feed = [x_conv]
    for l, c in zip(self.additional_blocks, self.extra_conv[1:]):
        x = l(x)
        x_conv = c(x)
        detection_feed.append(x_conv)
    
    # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
    locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

    # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
    return locs, confs

  def freeze(self, freeze_add=True):
    for name, param in self.named_parameters():
      if 'conf' in name or 'loc.' in name or 'merge' in name:
        param.requires_grad = True
      elif not freeze_add and 'additional' in name:
        param.requires_grad = True
      else:
        param.requires_grad = False

  def train_setbn(self, bn_eval=True):
    self.train()
    if bn_eval: self._eval_helper(next(self.modules()))

  def _eval_helper(self, module):
    if isinstance(module, torch.nn.BatchNorm2d): 
      module.eval()
    for child in module.children():
      self._eval_helper(child)
