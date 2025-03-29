import torch
from pathlib import Path

def save_checkpoint(model, optimizer, scheduler, filename, epoch, best_val_loss):
    """
    Save model checkpoint with additional training state
    """
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, filename)

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

def find_best_checkpoint(checkpoint_dir):
    """
    Find the checkpoint with the lowest validation loss in the given directory
    """
    checkpoints = list(Path(checkpoint_dir).glob("checkpoint_epoch_*.pth.tar"))
    if not checkpoints:
        return None
        
    # Extract validation loss from checkpoint filenames and find minimum
    min_loss_checkpoint = min(
        checkpoints,
        key=lambda x: float(str(x).split('valloss_')[1].split('.pth')[0])
    )
    return str(min_loss_checkpoint)

def cleanup_old_checkpoints(checkpoint_dir, keep_best_n=5):
    """
    Keep only the N best checkpoints (with lowest validation loss)
    """
    checkpoints = list(Path(checkpoint_dir).glob("checkpoint_epoch_*.pth.tar"))
    if len(checkpoints) > keep_best_n:
        # Sort checkpoints by validation loss (extracted from filename)
        checkpoints.sort(key=lambda x: float(str(x).split('valloss_')[1].split('.pth')[0]))
        # Remove the checkpoints with higher loss
        for checkpoint in checkpoints[keep_best_n:]:
            checkpoint.unlink()  # Delete the file