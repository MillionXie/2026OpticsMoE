"""Reuse the original paired renderer, supply raw instruction bytes only."""
import torch
from .models import tokenize_text

from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import (
    ExpandedUnifiedProductEditDataset,_load_rows,_catalogue,instruction_rows,TARGET_CATEGORIES,
    TRAIN_COMBINATIONS,TEST_COMBINATIONS,
)


class TextOnlyEditorDataset(ExpandedUnifiedProductEditDataset):
    def __init__(self,data_dir,split):
        self.sources=[r for r in _load_rows(data_dir,split) if r['category'] in TARGET_CATEGORIES]
        self.catalogue=_catalogue(data_dir)
        self.by_category={category:[(i,r) for i,r in enumerate(self.catalogue) if r['category']==category]
                          for category in TARGET_CATEGORIES}
        self.split=split;self.image_size=256
        self.instructions=instruction_rows(data_dir)
        if not self.sources or any(len(values)!=4 for values in self.by_category.values()):
            raise ValueError('Original three-category, four-design editor catalogue differs')
        tokenize_text([r['prompt'] for r in self.instructions])
        self.lookup={r['prompt']:i for i,r in enumerate(self.instructions)}
        self.text=torch.empty(len(self.instructions),0)
        self.scenes=TRAIN_COMBINATIONS if split=='train' else TEST_COMBINATIONS
