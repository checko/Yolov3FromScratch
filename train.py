import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from yolov3 import YOLOv3
from dataset import Dataset
from utils import (
    iou,
    save_checkpoint,
)
from const import (
    ANCHORS,
    s,
    device,
    leanring_rate,  # Note: there's a typo in 'learning_rate'
    batch_size,
    epochs,
    save_model
)
from dataset import train_transform

# Defining YOLO loss class 
class YOLOLoss(nn.Module): 
	def __init__(self): 
		super().__init__() 
		self.mse = nn.MSELoss() 
		self.bce = nn.BCEWithLogitsLoss() 
		self.cross_entropy = nn.CrossEntropyLoss() 
		self.sigmoid = nn.Sigmoid() 
	
	def forward(self, pred, target, anchors): 
		# Identifying which cells in target have objects 
		# and which have no objects 
		obj = target[..., 0] == 1
		no_obj = target[..., 0] == 0

		# Calculating No object loss 
		no_object_loss = self.bce( 
			(pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]), 
		) 

		
		# Reshaping anchors to match predictions 
		anchors = anchors.reshape(1, 3, 1, 1, 2) 
		# Box prediction confidence 
		box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), 
							torch.exp(pred[..., 3:5]) * anchors 
							],dim=-1) 
		# Calculating intersection over union for prediction and target 
		ious = iou(box_preds[obj], target[..., 1:5][obj]).detach() 
		# Calculating Object loss 
		object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), 
							ious * target[..., 0:1][obj]) 

		
		# Predicted box coordinates 
		pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) 
		# Target box coordinates 
		target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors) 
		# Calculating box coordinate loss 
		box_loss = self.mse(pred[..., 1:5][obj], 
							target[..., 1:5][obj]) 

		
		# Claculating class loss 
		class_loss = self.cross_entropy((pred[..., 5:][obj]), 
								target[..., 5][obj].long()) 

		# Total loss 
		return ( 
			box_loss 
			+ object_loss 
			+ no_object_loss 
			+ class_loss 
		)


# Change the training_loop function definition
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch):
    # Creating a progress bar 
    progress_bar = tqdm(loader, leave=True) 
    
    # Initializing a list to store the losses 
    losses = [] 
    
    # Keep track of batch number for tensorboard
    batch_idx = 0
    
    # Iterating over the training data 
    for _, (x, y) in enumerate(progress_bar): 
        x = x.to(device) 
        y0, y1, y2 = ( 
            y[0].to(device), 
            y[1].to(device), 
            y[2].to(device), 
        ) 

        with torch.cuda.amp.autocast(): 
            # Getting the model predictions 
            outputs = model(x) 
            # Calculating the loss at each scale 
            loss = ( 
                loss_fn(outputs[0], y0, scaled_anchors[0]) 
                + loss_fn(outputs[1], y1, scaled_anchors[1]) 
                + loss_fn(outputs[2], y2, scaled_anchors[2]) 
            ) 

        # Add the loss to the list 
        losses.append(loss.item()) 

        # Log to tensorboard (every batch)
        writer.add_scalar('Loss/train', loss.item(), batch_idx)
        
        # Reset gradients 
        optimizer.zero_grad() 

        # Backpropagate the loss 
        scaler.scale(loss).backward() 

        # Optimization step 
        scaler.step(optimizer) 

        # Update the scaler for next iteration 
        scaler.update() 

        # update progress bar with loss 
        mean_loss = sum(losses) / len(losses) 
        progress_bar.set_postfix(loss=mean_loss)
        
        batch_idx += 1
    
    # Log epoch mean loss
    epoch_loss = sum(losses) / len(losses)
    writer.add_scalar('Loss/epoch', epoch_loss, epoch)
    
    return epoch_loss


# Creating the model from YOLOv3 class 
model = YOLOv3().to(device) 

# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 

# Defining the loss function 
loss_fn = YOLOLoss() 

# Defining the scaler for mixed precision training 
scaler = torch.cuda.amp.GradScaler() 

# Defining the train dataset 
train_dataset = Dataset( 
    txt_file="pascal_voc/ImageSets/Main/train.txt",
    image_dir="pascal_voc/JPEGImages/",
    label_dir="pascal_voc/Annotations/",
    anchors=ANCHORS, 
    transform=train_transform 
) 

# Defining the train data loader 
train_loader = torch.utils.data.DataLoader( 
	train_dataset, 
	batch_size = batch_size, 
	num_workers = 2, 
	shuffle = True, 
	pin_memory = True, 
) 

# Scaling the anchors 
scaled_anchors = ( 
	torch.tensor(ANCHORS) *
	torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
).to(device) 

# Initialize TensorBoard writer
writer = SummaryWriter('runs/yolov3_training')

# Modify the training loop where the function is called
# Training the model 
for e in range(1, epochs+1): 
	print("Epoch:", e) 
	training_loop(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scaler=scaler,
        scaled_anchors=scaled_anchors,
        epoch=e  # Pass the epoch number
    )

	# Saving the model 
	if save_model: 
		save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
