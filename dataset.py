import torch
import numpy as np
import os
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import iou, convert_cells_to_bboxes, nms, plot_image
from const import ANCHORS, image_size, s, class_labels
import xml.etree.ElementTree as ET



class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, txt_file, image_dir, label_dir, anchors,
        image_size=416, grid_sizes=[13, 26, 52],
        num_classes=20, transform=None
    ):
        with open(txt_file, 'r') as file:
            self.image_list = [line.strip() for line in file.readlines()]
        
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.grid_sizes = grid_sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        
        img_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{image_name}.png")
            
        label_path = os.path.join(self.label_dir, f"{image_name}.xml")
        
        try:
            bboxes = self.parse_xml_annotation(label_path)
        except Exception as e:
            print(f"Error loading annotation for {image_name}: {e}")
            bboxes = []

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6))
                for s in self.grid_sizes]
        
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]),
                            self.anchors,
                            is_pred=False)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                
                s = self.grid_sizes[scale_idx]
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = s * x - j, s * y - i
                    width_cell, height_cell = (width * s, height * s)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)

    def parse_xml_annotation(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size for normalization
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)
        
        boxes = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            # Convert class name to index based on class_labels
            class_idx = class_labels.index(class_name)
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (x_center, y_center, width, height)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            boxes.append([x_center, y_center, width, height, class_idx])
            
        return boxes


# Training transforms with augmentation
train_transform = A.Compose([
    A.LongestMaxSize(max_size=image_size),
    A.PadIfNeeded(
        min_height=image_size,
        min_width=image_size,
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255
    ),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format="yolo",
    min_visibility=0.4,
    label_fields=[]
))

# Validation transforms without augmentation
val_transform = A.Compose([
    A.LongestMaxSize(max_size=image_size),
    A.PadIfNeeded(
        min_height=image_size,
        min_width=image_size,
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255
    ),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format="yolo",
    min_visibility=0.4,
    label_fields=[]
))

if __name__ == "__main__":
    dataset = Dataset(
        txt_file="pascal_voc/VOC2012/ImageSets/Main/train.txt",
        image_dir="pascal_voc/VOC2012/JPEGImages/",
        label_dir="pascal_voc/VOC2012/Annotations/",
        grid_sizes=[13, 26, 52],
        anchors=ANCHORS,
        transform=test_transform
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )

    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor([13, 26, 52] ).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    x, y = next(iter(loader))
    boxes = []
    for i in range(y[0].shape[1]):
        anchor = scaled_anchors[i]
        boxes += convert_cells_to_bboxes(
            y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor
        )[0]

    boxes = nms(boxes, iou_threshold=1, threshold=0.7)
    plot_image(x[0].permute(1,2,0).to("cpu"), boxes)
