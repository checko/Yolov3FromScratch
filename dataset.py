import torch
import numpy as np
import os
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
IMAGE_SIZE = 416
GRID_SIZE = [13, 26, 52]

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
            
        label_path = os.path.join(self.label_dir, f"{image_name}.txt")
        
        try:
            bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
            bboxes = np.roll(bboxes, -1, axis=1).tolist()
        except:
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


train_transform = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(
        min_height=IMAGE_SIZE,
        min_width=IMAGE_SIZE,
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5,
        p=0.5
    ),
    A.HorizontalFlip(p=0.5),
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

test_transform = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(
        min_height=IMAGE_SIZE,
        min_width=IMAGE_SIZE,
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
        txt_file="train.txt",
        image_dir="images/",
        label_dir="labels/",
        grid_sizes=GRID_SIZE,
        anchors=ANCHORS,
        transform=test_transform
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )

    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
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
