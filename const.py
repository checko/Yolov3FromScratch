import torch
import os

# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable 
load_model = False
save_model = True

# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 

# Batch size for training 
batch_size = 24

# Learning rate for training 
leanring_rate = 1e-5

# Number of epochs for training 
epochs = 330

# Image size 
image_size = 416

# Grid cell sizes 
s = [image_size // 32, image_size // 16, image_size // 8] 

def load_class_labels(file_path='/home/charles-chang/datasets/coco/official/pascal_coco/classes.txt'):
    """Load class labels from a text file. If file doesn't exist, use default PASCAL VOC classes"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        # Default PASCAL VOC classes as fallback
        return [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

# Class labels 
class_labels = load_class_labels()

# Number of classes derived from class_labels
num_classes = len(class_labels)

# Training dropout rate
TRAIN_DROPOUT_RATE = 0.5  # Moved from train.py
