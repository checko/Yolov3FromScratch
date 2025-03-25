# YOLOv3 Implementation in PyTorch

This project is based on the YOLOv3 implementation tutorial from GeeksForGeeks ([YOLOv3 from Scratch using PyTorch](https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/)) with additional modifications and improvements to make it fully functional.

## Changes and Additions

1. Added missing utility functions:
   - `iou()` - Intersection over Union calculation
   - `convert_cells_to_bboxes()` - Convert cell predictions to bounding boxes
   - `nms()` - Non-maximum suppression
   - `plot_image()` - Visualization function for bounding boxes

2. Modified dataset handling:
   - Support for PASCAL VOC format
   - Added data augmentation using Albumentations
   - Fixed bounding box transformations

3. Enhanced model architecture:
   - Complete implementation of all YOLOv3 layers
   - Added proper skip connections
   - Fixed scale predictions

## Requirements

- Python 3.8+
- PyTorch 1.7+
- OpenCV
- Albumentations
- NumPy
- Pillow
- TensorBoard

## Directory Structure

```
myyolov3/
├── dataset.py      # Dataset and data loading utilities
├── yolov3.py       # YOLOv3 model implementation
├── utils.py        # Helper functions
├── train.py        # Training script
├── const.py        # Constants and configurations
├── runs/           # TensorBoard logs
└── README.md       # Project documentation
```

## Usage

1. Prepare your dataset in PASCAL VOC format
2. Update the paths in main script
3. Run training:
```bash
python train.py
```

4. Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=runs
```
Then open http://localhost:6006 in your web browser to view training metrics.

## Credits

- Original tutorial by GeeksForGeeks
- YOLOv3 paper: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)