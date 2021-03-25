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


class SpatialAveragePrecision(Metric):
    def __init__(
        self,
        update_fun,
        compute_fun,
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
        self.update_fun = update_fun
        self.compute_fun = compute_fun

    def update(
        self,
        preds: LongTensor,
        target: LongTensor,
        pred_classes: Optional[LongTensor] = None,
        target_classes: Optional[LongTensor] = None,
        pred_grouping: Optional[LongTensor] = None,
        target_grouping: Optional[LongTensor] = None,
    ):
        output_preds, output_target = self.update_fun(
            preds = preds,
            target = target,
            pred_classes = pred_classes,
            target_classes = target_classes,
            pred_grouping = pred_grouping,
            target_grouping = target_grouping,
            num_classes = self.num_classes,
        )
        self.append_states(output_preds, output_target)

    def append_states(self, preds, target):
        for k,v in preds.items():
            attr_name = f'preds_{k}'
            if not hasattr(self, attr_name):
                self.add_state(
                    attr_name,
                    default = [],
                    dist_reduce_fx = None)

            getattr(self, attr_name).append(v)

        for k,v in target.items():
            attr_name = f'target_{k}'
            if not hasattr(self, attr_name):
                self.add_state(
                    attr_name,
                    default = [],
                    dist_reduce_fx = None)

            getattr(self, attr_name).append(v)
        
    def compute(self):
        pred_attrs = [x for x in dir(self) if 'preds' in x]
        target_attrs = [x for x in dir(self) if 'target' in x]

        preds = {k.replace('preds_', ''):torch.cat(getattr(self, k)) for k in pred_attrs}
        target = {k.replace('target_', ''):torch.cat(getattr(self, k)) for k in target_attrs}

        return self.compute_fun(preds, target, self.num_classes)


class BoxAveragePrecision(SpatialAveragePrecision):
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
            num_classes=num_classes,
            update_fun = _box_average_precision_update,
            compute_fun = _box_average_precision_compute,
        )


    def update(
        self,
        preds: LongTensor,
        target: LongTensor,
        pred_classes: Optional[LongTensor] = None,
        target_classes: Optional[LongTensor] = None,
        pred_grouping: Optional[LongTensor] = None,
        target_grouping: Optional[LongTensor] = None,
    ):
        super().update(
            preds,
            target,
            pred_classes,
            target_classes,
            pred_grouping,
            target_grouping)
        
    def compute(self):
        return super().compute()


class MaskAveragePrecision(SpatialAveragePrecision):
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
            num_classes=num_classes,
            update_fun = _mask_average_precision_update,
            compute_fun = _mask_average_precision_compute,
        )


    def update(
        self,
        preds: LongTensor,
        target: LongTensor,
        pred_classes: Optional[LongTensor] = None,
        target_classes: Optional[LongTensor] = None,
        pred_grouping: Optional[LongTensor] = None,
        target_grouping: Optional[LongTensor] = None,
    ):
        super().update(
            preds,
            target,
            pred_classes,
            target_classes,
            pred_grouping,
            target_grouping)
        
    def compute(self):
        return super().compute()

# TODO:
# - Implement AP at threshold

preds = torch.randint(0, 2000, [40, 4])
preds[...,2:] = preds[...,:2] + torch.randint(100, 500, [40, 2])

target = torch.randint(0, 2000, [20, 4])
target[...,2:] = target[...,:2] + torch.randint(100, 500, [20, 2])

bap = BoxAveragePrecision()
bap.update(preds, target)
bap.update(preds, target)
bap.update(preds, target)

print(bap.compute())

map = MaskAveragePrecision()

preds = torch.randint(0, 2, [40, 224, 224])
target = torch.randint(0, 2, [20, 224, 224])

map.update(preds, target)
map.update(preds, target)
map.update(preds, target)

print(map.compute())

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