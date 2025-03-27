import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from yolov3 import YOLOv3
from dataset import Dataset, train_transform, val_transform  # Update import
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
from loss import YOLOLoss
import os
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def validation_loop(loader, model, loss_fn, scaled_anchors):
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y0, y1, y2 = (
                y[0].to(device),
                y[1].to(device),
                y[2].to(device),
            )
            
            outputs = model(x)
            loss = (
                loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )
            val_losses.append(loss.item())
            
    model.train()
    return sum(val_losses) / len(val_losses)

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

# Create checkpoint directory
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 

# Add the scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Number of epochs before first restart
    T_mult=2,  # Multiply T_0 by this factor after each restart
    eta_min=1e-6  # Minimum learning rate
)

# Defining the loss function 
loss_fn = YOLOLoss() 

# Defining the scaler for mixed precision training 
scaler = torch.cuda.amp.GradScaler() 

# Defining the train dataset 
train_dataset = Dataset( 
    txt_file="pascal_voc/VOC2012/ImageSets/Main/train.txt",
    image_dir="pascal_voc/VOC2012/JPEGImages/",
    label_dir="pascal_voc/VOC2012/Annotations/",
    anchors=ANCHORS, 
    transform=train_transform 
) 

val_dataset = Dataset(
    txt_file="pascal_voc/VOC2012/ImageSets/Main/val.txt",
    image_dir="pascal_voc/VOC2012/JPEGImages/",
    label_dir="pascal_voc/VOC2012/Annotations/",
    anchors=ANCHORS,
    transform=val_transform  # Use validation transform instead of test transform
)

# Defining the train data loader 
train_loader = torch.utils.data.DataLoader( 
	train_dataset, 
	batch_size = batch_size, 
	num_workers = 2, 
	shuffle = True, 
	pin_memory = True, 
) 

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
)

# Scaling the anchors 
scaled_anchors = ( 
	torch.tensor(ANCHORS) *
	torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
).to(device) 

# Initialize TensorBoard writer
writer = SummaryWriter('runs/yolov3_training')

# Track best validation loss
best_val_loss = float('inf')

# Training loop with validation
for e in range(1, epochs+1):
    print(f"Epoch: {e}")
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Training
    train_loss = training_loop(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scaler=scaler,
        scaled_anchors=scaled_anchors,
        epoch=e
    )
    
    # Validation
    val_loss = validation_loop(
        loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        scaled_anchors=scaled_anchors
    )
    
    # Log metrics
    writer.add_scalar('Loss/val', val_loss, e)
    writer.add_scalar('Loss/train_val_ratio', train_loss/val_loss, e)
    writer.add_scalar('Learning_rate', current_lr, e)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
    
    # Step the scheduler
    scheduler.step()
    
    # Save checkpoint logic
    if save_model and val_loss < best_val_loss:
        checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{e}_valloss_{val_loss:.4f}.pth.tar"
        # Include scheduler state in checkpoint
        save_checkpoint(model, optimizer, scheduler, filename=str(checkpoint_file))
        best_val_loss = val_loss
