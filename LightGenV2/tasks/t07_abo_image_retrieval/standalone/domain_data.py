"""Training-only domain expansion. Original test and 120-product gallery stay fixed."""
import math
from collections import defaultdict
from dataclasses import replace


def paired_view_indices(samples, groups, indices, rng):
    """Different image of the SAME training product; never a same-class substitute."""
    paired=[]
    for i in indices:
        sample=samples[i]
        candidates=[j for j in groups[sample.category_id][sample.product_id]
                    if samples[j].image_path != sample.image_path]
        if not candidates:raise ValueError(f'No independent view for product {sample.product_id}')
        paired.append(rng.choice(candidates))
    return paired


def view_consistency_loss(first, second):
    """Symmetric cosine alignment; both views receive gradients, no inference head."""
    import torch.nn.functional as F
    if first.shape!=second.shape or first.ndim!=2:
        raise ValueError('Paired feature shapes must match [batch, dimension]')
    return (1-F.cosine_similarity(first.float(),second.float(),dim=-1)).mean()


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


def epoch_batches(samples,target_count,mode,epoch,pretrain_epochs,minimum_steps,rng,target_products_per_class=2):
    """Cycle products without replacement within each category/domain, vary views.

    40 images/batch: 4 products/category; mixed mode defaults to 2 original + 2 external.
    Every active external product is visited at least once per epoch. Original
    products are oversampled deliberately, with measured exposure logged by trainer.
    """
    if type(target_products_per_class) is not int or not 1<=target_products_per_class<=3:
        raise ValueError('Mixed batches require 1, 2 or 3 original products per category')
    phase='external' if mode=='curriculum' and epoch<=pretrain_epochs else ('target' if mode=='target_control' else 'mixed')
    domains=defaultdict(lambda:defaultdict(list))
    for i,s in enumerate(samples):domains[(s.category_id,'target' if i<target_count else 'external')][s.product_id].append(i)
    categories=sorted({s.category_id for s in samples})
    quotas={'target':4} if phase=='target' else {'external':4} if phase=='external' else {'target':target_products_per_class,'external':4-target_products_per_class}
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
