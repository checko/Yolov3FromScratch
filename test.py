import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from dataset import Dataset, val_transform
from yolov3 import YOLOv3
from loss import YOLOLoss
from utils import (
    load_checkpoint,
    convert_cells_to_bboxes,
    nms,
    plot_image
)
from const import (
    ANCHORS,
    s,
    device,
    leanring_rate,  # Note: there's a typo in 'learning_rate'
    checkpoint_file
)

# Taking a sample image and testing the model 

# Setting the load_model to True 
load_model = True

# Defining the model, optimizer, loss function and scaler 
model = YOLOv3().to(device) 
optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 
loss_fn = YOLOLoss() 
scaler = torch.cuda.amp.GradScaler() 

# Loading the checkpoint 
if load_model: 
	load_checkpoint(checkpoint_file, model, optimizer, leanring_rate) 

# Defining the test dataset and data loader 
test_dataset = Dataset( 
    txt_file="pascal_voc/VOC2007/ImageSets/Main/test.txt",
    image_dir="pascal_voc/VOC2007/JPEGImages/",
    label_dir="pascal_voc/VOC2007/Annotations/",
    anchors=ANCHORS, 
    transform=val_transform 
) 
test_loader = torch.utils.data.DataLoader( 
	test_dataset, 
	batch_size = 1, 
	num_workers = 2, 
	shuffle = True, 
) 

# Getting a sample image from the test data loader 
x, y = next(iter(test_loader)) 
x = x.to(device) 

model.eval() 
with torch.no_grad(): 
	# Getting the model predictions 
	output = model(x) 
	# Getting the bounding boxes from the predictions 
	bboxes = [[] for _ in range(x.shape[0])] 
	anchors = ( 
			torch.tensor(ANCHORS) 
				* torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
			).to(device) 

	# Getting bounding boxes for each scale 
	for i in range(3): 
		batch_size, A, S, _, _ = output[i].shape 
		anchor = anchors[i] 
		boxes_scale_i = convert_cells_to_bboxes( 
							output[i], anchor, s=S, is_predictions=True
						) 
		for idx, (box) in enumerate(boxes_scale_i): 
			bboxes[idx] += box 
model.train() 

# Plotting the image with bounding boxes for each image in the batch 
for i in range(batch_size): 
	# Applying non-max suppression to remove overlapping bounding boxes 
	nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6) 
	# Plotting the image with bounding boxes 
	plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
