import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def iou(box1_wh, anchors, is_pred=True):
    """
    Calculate IoU between box1 and multiple anchors
    Args:
        box1_wh: tensor [width, height] of first box
        anchors: tensor [N, 2] of N anchor boxes (width, height)
        is_pred: boolean to indicate if box1 is prediction
    """
    if is_pred:
        box1_wh = box1_wh.exp()
        
    # Get area of box1
    box1_area = box1_wh[0] * box1_wh[1]
    
    # Get area of each anchor box
    anchor_area = anchors[:, 0] * anchors[:, 1]
    
    # Calculate intersection area
    intersection = torch.min(box1_wh[0], anchors[:, 0]) * torch.min(box1_wh[1], anchors[:, 1])
    
    # Calculate IoU
    iou = intersection / (box1_area + anchor_area - intersection + 1e-6)
    return iou

def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    """
    Convert cell predictions to bounding boxes
    Args:
        predictions: tensor [batch_size, num_anchors, S, S, num_classes+5]
        anchors: tensor [num_anchors, 2]
        s: grid size
        is_predictions: boolean to indicate if input is model prediction
    """
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
    
    cell_indices = (
        torch.arange(s)
        .repeat(batch_size, num_anchors, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / s * box_predictions[..., 2:4]
    
    converted_bboxes = torch.cat((scores, best_class, x, y, w_h), dim=-1).reshape(batch_size, num_anchors * s * s, 6)
    return converted_bboxes.tolist()

def nms(bboxes, iou_threshold, threshold):
    """
    Non-maximum suppression
    Args:
        bboxes: list [[score, class, x, y, w, h], ...]
        iou_threshold: threshold for IoU to remove overlapping boxes
        threshold: confidence threshold to filter weak detections
    """
    bboxes = [box for box in bboxes if box[0] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[1] != chosen_box[1]
            or box_iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])
            ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms

def plot_image(image, boxes):
    """
    Plot image with bounding boxes
    Args:
        image: tensor [H, W, C]
        boxes: list [[score, class, x, y, w, h], ...]
    """
    cmap = plt.get_cmap('tab20b')
    class_labels = ['person', 'car', 'dog']  # Add your class labels here
    
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box in boxes:
        score = box[0]
        class_pred = box[1]
        box = box[2:]
        upper_left_x = box[0] - box[2]/2
        upper_left_y = box[1] - box[3]/2
        
        rect = patches.Rectangle(
            (upper_left_x * 416, upper_left_y * 416),
            box[2] * 416,
            box[3] * 416,
            linewidth=2,
            edgecolor=cmap(class_pred * 4),
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            upper_left_x * 416,
            upper_left_y * 416,
            s=f"{class_labels[int(class_pred)]} {score:.1f}",
            color="white",
            verticalalignment="top",
            bbox={"color": cmap(class_pred * 4), "pad": 0},
        )
    
    plt.show()