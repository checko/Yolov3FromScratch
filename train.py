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
    save_model,
    num_classes  # Import num_classes from const
)
from loss import YOLOLoss
import os
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add this after your imports
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint and return the epoch and best validation loss
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    return epoch, best_val_loss

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
model = YOLOv3(num_classes=num_classes, dropout_rate=0.1).to(device) 

# Create checkpoint directory
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Defining the optimizer 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4  # L2 regularization
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5,
    verbose=True
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

# Add this before the training loop
start_epoch = 1
checkpoint_path = None  # Set this to your checkpoint path if you want to resume training

if checkpoint_path and os.path.exists(checkpoint_path):
    start_epoch, best_val_loss = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler
    )
    print(f"Resuming training from epoch {start_epoch} with validation loss {best_val_loss:.4f}")
else:
    best_val_loss = float('inf')

# Modify your training loop to start from start_epoch
for e in range(start_epoch, epochs+1):
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
        save_checkpoint(
            model, 
            optimizer, 
            scheduler, 
            filename=str(checkpoint_file),
            epoch=e,
            best_val_loss=val_loss
        )
        best_val_loss = val_loss
        
        # Keep only the 5 best checkpoints (lowest validation loss)
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth.tar"))
        if len(checkpoints) > 5:
            # Sort checkpoints by validation loss (extracted from filename)
            checkpoints.sort(key=lambda x: float(str(x).split('valloss_')[1].split('.pth')[0]))
            # Remove the checkpoints with higher loss
            for checkpoint in checkpoints[5:]:
                checkpoint.unlink()  # Delete the file
