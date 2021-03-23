
from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.functional.classification.average_precision import (
    _average_precision_update,
    _average_precision_compute,
)

def bin_mask_iou(pred_masks: Tensor, target_masks: Tensor) -> Tensor:
    """
    Calculates all pairwise IOU on binary masks

    Parameters
    ----------
    pred_masks : Tensor
        Predicted binary masks, should be a tensor of shape [p,i,j] or [p,c,i,j]
    target_masks : Tensor
        Target binary masks, should be a tensor of shape [t,i,j] or [t,c,i,j]

    Details
    -------
    The expected tensors can be either of dimensions [NUM_MASK, HEIGHT, WIDTH]
    or [NUM_MASK, NUM_CLASSES, HEIGHT, WIDTH], it assumes that the height, width
    and optionally the number of classes is the same between the two inputs.

    Returns
    -------
    Tensor
        A tensor with all pairwise IOUs, will have shape [p, t] if no classes
        are present in the input tensors or [c,p,t] if classes are.
    """
    # Should I assert that the input is actually 0-1 ??
    intersections = torch.einsum("p...ij, t...ij -> ...pt", pred_masks, target_masks)
    except_1 = torch.einsum("p...ij, t...ij -> ...pt", 1 - pred_masks, target_masks)
    except_2 = torch.einsum("p...ij, t...ij -> ...pt", pred_masks, 1 - target_masks)

    unions = intersections + except_1 + except_2

    iou = intersections / unions
    return iou


# Implementation from https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments
    ---------
    boxes: Tensor[N, 4]
        boxes for which the area will be computed. They
        are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications from https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments
    ---------
    boxes1: Tensor
        Tensor[N, 4]
    boxes2: Tensor
        Tensor[M, 4]

    Returns:
    Tensor:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def _get_mappings(iou_mat: Tensor) -> Tensor:
    """
    Get candidate pairs from a matrix of IntersectionOverUnions

    # From https://www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation?scriptVersionId=40816383

    >>> square = torch.zeros([1, 5, 5])
    >>> square[..., 1:, 1:] = 1
    >>> squares = torch.cat([square, torch.flip(square, (-1, -2)), torch.flip(square, (-1, ))])
    >>> iou = bin_mask_iou(squares.long(), torch.flip(squares.long(), (0,)))
    >>> _get_mappings(iou)
    tensor([[-1., -1.,  2.],
            [-1.,  1., -1.],
            [ 0., -1., -1.]])
    """
    pred_count, target_count = iou_mat.shape
    mappings = torch.zeros_like(iou_mat) - 1

    #first mapping (max iou for first pred_box)
    if not iou_mat[0,:].eq(0.).all():
        position = iou_mat[0,:].argsort()[-1]
        # if not a zero column
        mappings[0,position] = position

    for pred_id in range(1,pred_count):
        # Sum of all the previous mapping columns will let 
        # us know which gt-boxes are already assigned
        not_assigned = torch.logical_not(mappings[:pred_id,:].add(1).sum(0)).long()

        # Considering unassigned gt-boxes for further evaluation 
        targets = not_assigned * iou_mat[pred_id,:]

        # If no gt-box satisfy the previous conditions
        # for the current pred-box, ignore it (False Positive)
        if targets.eq(0).all():
            continue

        # max-iou from current column after all the filtering
        # will be the pivot element for mapping
        pivot = targets.argsort()[-1]
        mappings[pred_id, pivot] = pivot

    return mappings.clone()


def _map_predictions(ious: Tensor) -> Tuple[Tensor, Tensor]:
    mappings = _get_mappings(ious)

    tps_map = mappings.max(axis = 0)

    subset_index = torch.ones(ious.shape[0]).bool()
    subset_index[tps_map.indices[tps_map.values >= 0]] = False

    tps_vals = ious[~subset_index, ].sum(axis = 1)
    fps_vals = torch.ones(subset_index.sum()) * float('inf')
    fns_vals = torch.ones_like(tps_map.indices[tps_map.values < 0]) * -float('inf')

    preds_out = torch.cat([tps_vals, fps_vals , fns_vals])
    target_out = torch.cat([
        torch.ones_like(tps_vals),
        -torch.ones_like(fps_vals),
        torch.ones_like(fns_vals)])

    return preds_out, target_out


def _binary_mask_average_precision_update(
    preds: Tensor,
    target: Tensor,
) -> Tuple[Tensor, Tensor, int, int]:
    ious = bin_mask_iou(pred_masks=preds, target_masks=target)
    preds_in, target_in = _map_predictions(ious)

    return _average_precision_update(preds_in, target_in, pos_label = 1)


def _binary_mask_average_precision_compute(*args, **kwargs) -> Tensor:
    # masks are converted by _binary_mask_average_precision_update to preds and
    # targets that are compatible with the standard avg_precision_compute
    return _average_precision_compute(*args, **kwargs)


def _binary_box_average_precision_update(
    preds: Tensor,
    targets: Tensor,
) -> Tuple[Tensor, Tensor, int, int]:
    ious = box_iou(preds, targets)
    preds_in, target_in = _map_predictions(ious)

    return _average_precision_update(preds_in, target_in, pos_label = 1)


def _binary_box_average_precision_compute(*args, **kwargs):
    # boxes are converted by _binary_box_average_precision_update to preds and
    # targets that are compatible with the standard avg_precision_compute
    return _average_precision_compute(*args, **kwargs)


def binary_box_average_precision(
    preds: Tensor,
    target: Tensor,
) -> Tensor:
    """
    Calculates the average precision of a series of bounding boxes

    Parameters
    ----------
    preds : Tensor
        Predicted bounding boxes, should be a tensor of shape [b,4]
    target : Tensor
        Target bounding boxes, should be a tensor of shape [p,4]

    Details
    -------
    Each box is applied maximum to 1 target. With this constrain it calculates
    the precision-recall curve of the boxes, using the intersection-over-union.

    For further detail check:
    TODO add here a link explaining the metric ...


    Returns
    -------
    Tensor of size 1 with the average precision.
    """
    preds, target, num_classes, pos_label = _binary_box_average_precision_update(preds, target,)
    return _binary_box_average_precision_compute(preds, target, num_classes, pos_label, sample_weights = None)


def binary_mask_average_precision(
    preds: Tensor,
    target: Tensor,
):
    """
    Calculates the average precision of a series of binary masks

    Parameters
    ----------
    preds : Tensor
        Predicted masks with values 1 and 0, should be of size [NUM_PREDS, H, W]
    target : Tensor
        Target masks with values 1 and 0, should be of size [NUM_TARGETS, H, W]

    Returns
    -------
    Tensor

    Example
    -------
    >>> preds = torch.randint(0, 2, size = [15, 224, 224])
    >>> target = torch.randint(0, 2, size = [5, 224, 224])
    >>> binary_mask_average_precision(preds, target)
    """
    preds, target, num_classes, pos_label = _binary_mask_average_precision_update(preds, target,)
    return _binary_mask_average_precision_compute(preds, target, num_classes, pos_label, sample_weights = None)


def test_binary_mask_iou_works():
    square = torch.zeros([1, 5, 5])
    square[..., 1:, 1:] = 1

    squares = torch.cat([square, torch.flip(square, (-1, -2))])
    squares
    # tensor([[[0., 0., 0., 0., 0.],
    #          [0., 1., 1., 1., 1.],
    #          [0., 1., 1., 1., 1.],
    #          [0., 1., 1., 1., 1.],
    #          [0., 1., 1., 1., 1.]],
    #
    #         [[1., 1., 1., 1., 0.],
    #          [1., 1., 1., 1., 0.],
    #          [1., 1., 1., 1., 0.],
    #          [1., 1., 1., 1., 0.],
    #          [0., 0., 0., 0., 0.]]])

    out = bin_mask_iou(squares.long(), squares.long())
    expected_out = torch.tensor([[1.0000, 0.3913], [0.3913, 1.0000]])
    assert out.shape == torch.Size([2, 2])
    assert torch.all((out - expected_out) < 1e-4)

    stacked_squares = torch.stack([squares] * 4, dim=0)
    out = bin_mask_iou(stacked_squares, stacked_squares)
    assert out.shape == torch.Size([2, 4, 4])
    assert torch.all(out == 1.0)
    
    preds, target = stacked_squares, torch.flip(stacked_squares, (-1,))
    out = bin_mask_iou(preds, target)
    assert torch.all((out -0.6) < 1e-4)

    stacked_squares = torch.stack([squares] * 4, dim=1)
    out = bin_mask_iou(stacked_squares, stacked_squares)
    assert out.shape == torch.Size([4, 2, 2])


def test_bin_box_mAP():
    pred_boxes = [
        [0, 0, 100, 100],
        [100,100,150,150],
        [300,300,450,450],
        [200,200,250,250],
    ]
    target_boxes = [
        [10, 10, 110, 110],
        [110,110,150,150],
        [1000, 1000, 1100, 1100],
    ]
    
    preds, targets = torch.tensor(pred_boxes), torch.tensor(target_boxes)
    outs = binary_box_average_precision(preds, targets)
    assert torch.all(outs < 1)
    assert len(outs.flatten()) == 1

    preds, targets = torch.tensor(pred_boxes[:-2]), torch.tensor(target_boxes)
    outs = binary_box_average_precision(preds, targets)
    assert torch.all(outs - 1 < 0.001)
    assert len(outs.flatten()) == 1

    preds, targets = torch.tensor(pred_boxes[2:]), torch.tensor(target_boxes)
    outs = binary_box_average_precision(preds, targets)

def test_bin_mask_mAP():
    pred_masks = torch.randint(0,2,[15,55,55])
    target_masks = torch.randint(0,2,[5,55,55])
    outs = binary_mask_average_precision(pred_masks, target_masks)
    assert torch.all(outs < 1)
    assert len(outs.flatten()) == 1

    pred_mask = [
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ]
    ]
    pred_masks = torch.tensor([pred_mask]*5).squeeze()

    target_mask = [
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    ]
    
    preds, targets = torch.tensor(pred_masks), torch.tensor(target_mask)
    outs = binary_mask_average_precision(preds, targets)
    assert torch.all(outs < 1)
    assert len(outs.flatten()) == 1

    preds, targets = torch.tensor(pred_mask).flip(2), torch.tensor(target_mask)
    outs = binary_mask_average_precision(preds, targets)

    preds, targets = torch.tensor(pred_mask), torch.tensor(pred_mask)
    outs = binary_mask_average_precision(preds, targets)
    assert torch.all(outs - 1 < 1e-5)
    assert len(outs.flatten()) == 1