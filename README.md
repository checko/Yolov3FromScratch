# YOLOv3 Implementation in PyTorch

This project is based on the YOLOv3 implementation tutorial from GeeksForGeeks ([YOLOv3 from Scratch using PyTorch](https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/)) with additional modifications and improvements to make it fully functional.

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
├── loss.py         # Loss function implementation
├── train.py        # Training script
├── test.py         # Testing and visualization script
├── const.py        # Constants and configurations
├── runs/           # TensorBoard logs
└── README.md       # Project documentation
```

## Dataset Structure

```
pascal_voc/
├── VOC2012/
│   ├── Annotations/         # XML annotation files
│   ├── ImageSets/
│   │   └── Main/
│   │       └── train.txt    # Training image names
│   └── JPEGImages/         # Image files (.jpg)
│
└── VOC2007/
    ├── Annotations/         # XML annotation files
    ├── ImageSets/
    │   └── Main/
    │       └── test.txt     # Testing image names
    └── JPEGImages/         # Image files (.jpg)
```

## Usage

1. Prepare your PASCAL VOC datasets:
   - VOC2012 for training
   - VOC2007 for testing
2. Update the paths in training and testing scripts
3. Run training:
```bash
python train.py
```

4. Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=runs
```

5. Test the model using VOC2007:
```bash
python test.py
```

This will load a trained model checkpoint and display test images with predicted bounding boxes.

## Credits

- Original tutorial by GeeksForGeeks
- YOLOv3 paper: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)