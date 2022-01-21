# SSD-Flow

This code performs object detection on the Waymo Open Dataset for autonomous driving. It builds on top of the Single-Shot Detector model, integrating an optical flow network in order take advantage of the temporal relationships between consecutive frames.

The following repositories are required:
* https://github.com/waymo-research/waymo-open-dataset
* https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD/
* https://github.com/princeton-vl/RAFT
* https://github.com/Cartucho/mAP
