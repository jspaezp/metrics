from functional.classification.precision_recall_curve import precision_recall_curve
from typing import Optional, Any

import torch
from torch import Tensor, IntTensor, BoolTensor
from torchmetrics import AveragePrecision
from torchmetrics.functional.classification.spatial_average_precision import (
    _binary_mask_average_precision_update,
    _binary_mask_average_precision_compute,
    _binary_box_average_precision_update,
    _binary_box_average_precision_compute,
)


class BoxMeanAveragePrecision(AveragePrecision):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            pos_label=pos_label,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        raise NotImplementedError

    def update(
        self,
        preds: Tensor,
        target: Tensor,
        pred_grouping: Optional(LongTensor) = None,
        target_grouping: Optional(LongTensor) = None,
    ):
        # (preds: float = [N1,C] , targets: long/bool = [N2,C] , pred_boxes = [x,x,y,y], target_boxes = [x,x,y,y], pred_grouping: long = [N1], target_grouping: long = [N2], iou_cutoff = 0.5)
        # Logic to match the boxes
        if pred_grouping is not None or target_grouping is not None:
            raise NotImplementedError

        preds_in, target_in, num_classes, pos_label = _binary_box_average_precision_update(
            preds, target, self.num_classes, self.pos_label,
        )
        self.num_classes = num_classes
        self.pos_label = pos_label

        super().update(preds=preds_in, target=target_in)


def MaskMAP(
    preds: Tensor,
    targets: IntTensor,
    pred_masks: Tensor = None,
    target_masks: IntTensor = None,
    iou_threshold: float = 0.5,
    pred_grouping: Optional(Tensor) = None,
    target_grouping: Optional(Tensor) = None,
):
    raise NotImplementedError

class MaskAveragePrecisionAtThreshold():
    def __init__(self, num_classes, threshold = 0.5):
        raise NotImplementedError

    def update(self, preds, target):
        raise NotImplementedError

    def compute(self, preds, target):
        raise NotImplementedError


def BinMaskMAP(
    preds: Tensor,
    targets: IntTensor,
    pred_masks: Tensor = None,
    target_masks: IntTensor = None,
    iou_threshold: float = 0.5,
):
    ious = bin_mask_iou(pred_masks, target_masks)
    mappings = _get_mappings(ious)

    assert mappings.sum(1).le(1).all()
    assert mappings.sum(0).le(1).all()

    scaled_mappings = torch.einsum("p..., ...pt -> ...pt", preds, mappings)

    class_auroc = auroc(preds = ious, target = mappings.long(), num_classes=len(target_masks))
    # This would be the segmentation AP
    ious = ious.where(ious > iou_threshold, torch.tensor(0.))


    mapped_preds = torch.einsum("...pt -> ...p", scaled_mappings)
    mapped_targets = torch.einsum("...pt, t... -> ...p", torch.ones_like(mappings), targets)
    average_precision(preds = mapped_preds, target = mappings.sum(1).long().unsqueeze(0), num_classes = 1)


    _average_precision_update()
    tp = mappings.sum()
    fp = mappings.sum(0).eq(0).sum()
    fn = mappings.sum(1).eq(0).sum()

    mAP = tp / (tp+fp+fn)

    return mAP

from torchmetrics.functional import average_precision, auroc

square = torch.zeros([1, 5, 5])
square[..., 1:, 1:] = 1

squares = torch.cat([square, torch.flip(square, (-1, -2))])
squares

pred_masks = torch.randint(0,2,[15,55,55])
target_masks = torch.randint(0,2,[5,55,55])

preds = torch.rand([len(pred_masks),1])
targets = torch.ones([len(target_masks),1])

scaled_mappings[..., 0]

average_precision(scaled_mappings[0], targets, num_classes=1)


def get_candidates(preds, target, ):
    pass
