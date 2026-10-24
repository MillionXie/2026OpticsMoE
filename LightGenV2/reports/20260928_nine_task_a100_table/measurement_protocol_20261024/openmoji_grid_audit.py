"""CPU-only original grid metrics and exact native-BF16 edit decision contract."""
from __future__ import annotations
import numpy as np


def metrics(prediction,edited,task_prediction,row):
    source=np.asarray(row['source_grid']);target=np.asarray(row['target_grid'])
    prediction=np.asarray(prediction);edited=np.asarray(edited,dtype=bool)
    if source.shape!=(6,6) or target.shape!=(6,6) or prediction.shape!=(6,6) or edited.shape!=(6,6):
        raise ValueError('Original grid shapes differ')
    correct=prediction==target;changed=source!=target;foreground=target>0;preserve=~changed
    def mean(mask):return float(correct[mask].astype(np.float32).mean())
    intersection=int((edited & changed).sum());union=int((edited | changed).sum())
    actual={(int(prediction[i,j]),i,j) for i,j in zip(*np.nonzero(prediction>0))}
    gold={(int(target[i,j]),i,j) for i,j in zip(*np.nonzero(target>0))}
    match=len(actual & gold);precision=match/max(1,len(actual));recall=match/max(1,len(gold))
    return dict(cell_accuracy=float(correct.astype(np.float32).mean()),changed_cell_accuracy=mean(changed),
        foreground_category_accuracy=mean(foreground) if foreground.any() else 1.,preserved_cell_accuracy=mean(preserve),
        edit_grid_iou=intersection/union if union else 1.,
        object_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        scene_exact_match=float(np.array_equal(prediction,target)),task_accuracy=float(task_prediction==row['task_index']))


def native_bf16_edit(logits):
    """For finite BF16 values, sigmoid(x).to(BF16)>=.5 iff x>=-2**-8.

    This is ONLY an independent audit of the original quantized decision, not a
    replacement inference path. Formal workers always call original compose_prediction.
    Exhaustive finite-BF16 CPU validation is provided in the opt-in test below.
    """
    value=np.asarray(logits,dtype=np.float32)
    if not np.isfinite(value).all() or np.any(value.view(np.uint32)&np.uint32(0xffff)):
        raise ValueError('Expected actual finite BF16 logits saved losslessly as FP32')
    return value>=np.float32(-2**-8)
