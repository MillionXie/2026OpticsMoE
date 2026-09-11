"""Training-only domain expansion. Original test and 120-product gallery stay fixed."""
import math
from collections import defaultdict
from dataclasses import replace


def combine_training(target, external):
    train=[s for s in target if s.split=='train']
    blocked={s.product_id for s in target}
    names={s.category_id:s.category_name for s in target}
    if not external or any(s.product_id in blocked for s in external):
        raise ValueError('External pool is empty or overlaps protected target products')
    if any(names.get(s.category_id)!=s.category_name for s in external):
        raise ValueError('External labels do not match target category semantics')
    if {s.category_id for s in external}!=set(names):raise ValueError('Missing external category')
    result=train+[replace(s,split='train') for s in external]
    if len({s.sample_id for s in result})!=len(result):raise ValueError('Duplicate training sample ID')
    return result,len(train)


def epoch_batches(samples,target_count,mode,epoch,pretrain_epochs,minimum_steps,rng):
    """Cycle products without replacement within each category/domain, vary views.

    40 images/batch: 4 products/category; mixed mode uses 2 original + 2 external.
    Every active external product is visited at least once per epoch. Original
    products are oversampled deliberately, with measured exposure logged by trainer.
    """
    phase='external' if mode=='curriculum' and epoch<=pretrain_epochs else ('target' if mode=='target_control' else 'mixed')
    domains=defaultdict(lambda:defaultdict(list))
    for i,s in enumerate(samples):domains[(s.category_id,'target' if i<target_count else 'external')][s.product_id].append(i)
    categories=sorted({s.category_id for s in samples})
    quotas={'target':4} if phase=='target' else {'external':4} if phase=='external' else {'target':2,'external':2}
    if any(len(domains[(c,d)])<q for c in categories for d,q in quotas.items()):raise ValueError('Not enough distinct products for domain-balanced batches')
    steps=max(minimum_steps,max(math.ceil(len(domains[(c,d)])/q) for c in categories for d,q in quotas.items()))
    queues={}
    def take(key,count):
        picked=[]
        while len(picked)<count:
            if not queues.get(key):
                queues[key]=list(domains[key]);rng.shuffle(queues[key])
            pid=queues[key].pop()
            if pid not in picked:picked.append(pid)
        return [rng.choice(domains[key][pid]) for pid in picked]
    batches=[]
    for _ in range(steps):
        batch=[i for c in categories for d,q in quotas.items() for i in take((c,d),q)]
        rng.shuffle(batch);batches.append(batch)
    active=[i for i in range(len(samples)) if ('target' if i<target_count else 'external') in quotas]
    return phase,batches,active
