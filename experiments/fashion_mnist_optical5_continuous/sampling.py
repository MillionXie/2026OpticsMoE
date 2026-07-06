from __future__ import annotations

import random
from typing import Iterator,Sequence

from torch.utils.data import Sampler


class EpochClassMixedSampler(Sampler[int]):
    def __init__(self,indices:Sequence[int],labels:Sequence[int],num_classes:int,batch_size:int,seed:int,per_class_limit:int|None=None)->None:
        self.indices=list(map(int,indices));self.labels=list(map(int,labels));self.num_classes=num_classes;self.batch_size=batch_size;self.seed=seed;self.per_class_limit=per_class_limit;self.epoch=1;self.by_class={cls:[i for i in self.indices if self.labels[i]==cls] for cls in range(num_classes)}
        if any(not values for values in self.by_class.values()):raise ValueError("Every class must have training samples")
    def __len__(self):return sum(min(len(values),self.per_class_limit) if self.per_class_limit is not None else len(values) for values in self.by_class.values())
    def set_epoch(self,epoch:int)->None:self.epoch=int(epoch)
    def epoch_class_counts(self)->dict[int,int]:return {cls:min(len(values),self.per_class_limit) if self.per_class_limit is not None else len(values) for cls,values in self.by_class.items()}
    def __iter__(self)->Iterator[int]:
        rng=random.Random(self.seed+1_000_003*self.epoch);queues={cls:self._select(cls,rng) for cls in range(self.num_classes)};positions={cls:0 for cls in queues};remaining=sum(map(len,queues.values()))
        while remaining:
            batch=[]
            while len(batch)<self.batch_size and remaining:
                available=[cls for cls in queues if positions[cls]<len(queues[cls])];rng.shuffle(available)
                for cls in available:
                    if len(batch)>=self.batch_size:break
                    batch.append(queues[cls][positions[cls]]);positions[cls]+=1;remaining-=1
            yield from batch
    def _select(self,cls:int,rng:random.Random)->list[int]:
        source=list(self.by_class[cls]);fixed=random.Random(self.seed+97_409*(cls+1));fixed.shuffle(source)
        if self.per_class_limit is not None and self.per_class_limit<len(source):start=((self.epoch-1)*self.per_class_limit)%len(source);selected=[source[(start+i)%len(source)] for i in range(self.per_class_limit)]
        else:selected=source
        rng.shuffle(selected);return selected

