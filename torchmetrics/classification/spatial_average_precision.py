import copy
from functools import reduce
from typing import Optional, Any
from collections import defaultdict

import torch
from torch import Tensor, IntTensor, LongTensor
from torchmetrics import Metric
from torchmetrics.functional.classification.spatial_average_precision import (
    _mask_average_precision_update,
    _mask_average_precision_compute,
    _box_average_precision_update,
    _box_average_precision_compute,
)


class BoxAveragePrecision(Metric):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        self.add_state(
            "preds",
            default = defaultdict(lambda: list()),
            dist_reduce_fx=self.reduce_pred_dicts)

        self.add_state(
            "target",
            default = defaultdict(lambda: list()),
            dist_reduce_fx=self.reduce_pred_dicts)

    def update(
        self,
        preds: LongTensor,
        target: LongTensor,
        pred_classes: Optional[LongTensor] = None,
        target_classes: Optional[LongTensor] = None,
        pred_grouping: Optional[LongTensor] = None,
        target_grouping: Optional[LongTensor] = None,
    ):
        # (preds: float = [N1,C] , targets: long/bool = [N2,C] , pred_boxes = [x,x,y,y], target_boxes = [x,x,y,y], pred_grouping: long = [N1], target_grouping: long = [N2], iou_cutoff = 0.5)
        # Logic to match the boxes
        if pred_grouping is not None or target_grouping is not None:
            raise NotImplementedError

        output_preds, output_target = _box_average_precision_update(
            preds = preds,
            target = target,
            pred_classes = pred_classes,
            target_classes = target_classes,
            pred_grouping = pred_grouping,
            target_grouping = target_grouping,
            num_classes = self.num_classes,
        )

        for k,v in output_preds.items():
            self.preds[k].append(v)

        for k,v in output_target.items():
            self.target[k].append(v)
        
    def compute(self):
        preds = {k:torch.cat(x) for k,x in self.preds.items()}
        target = {k:torch.cat(x) for k,x in self.target.items()}

        return _box_average_precision_compute(preds, target, self.num_classes)

    @staticmethod
    def combine_pred_dict(target_dict, append_dict):
        target_dict = copy.deepcopy(target_dict)

        for k,v in append_dict.items():
            target_dict[k].append(v)

        return target_dict

    @staticmethod
    def reduce_pred_dicts(lst):
        return reduce(BoxAveragePrecision.combine_pred_dict, lst)


preds = torch.randint(0, 2000, [40, 4])
preds[...,2:] = preds[...,2:] + torch.randint(0, 500, [40, 2])

target = torch.randint(0, 2000, [20, 4])
target[...,2:] = target[...,2:] + torch.randint(0, 500, [20, 2])

bap = BoxAveragePrecision()
bap.update(preds, target)



"""

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
"""