"""Image-only preprocessing: no prompt, text tokens, attention mask or tokenizer."""
from concurrent.futures import ThreadPoolExecutor
import torch


def _prepare_cpu(loaded,images,settings):
    inputs=dict(loaded.processor(images=images,return_tensors='pt'))
    inputs={key:inputs[key] for key in ('pixel_values','image_grid_thw')}
    grid=inputs['image_grid_thw']
    if grid.shape!=(len(images),3) or not bool((grid[:,0]==1).all()):
        raise RuntimeError('Exactly one still image per example is required')
    if bool((grid.prod(1)>settings.max_visual_tokens).any()):
        raise RuntimeError('Visual token budget exceeded')
    if int(grid.prod(1).sum())!=len(inputs['pixel_values']):
        raise RuntimeError('Packed image tokens do not match the grid')
    return inputs


def _move(inputs,device):
    return {k:v.to(device,non_blocking=True) for k,v in inputs.items()}


def _prepare(loaded,images,settings):
    return _move(_prepare_cpu(loaded,images,settings),loaded.device)


def _prepared_batches(loader,loaded,settings):
    it=iter(loader)
    try:images,labels=next(it)
    except StopIteration:return
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='vision-preprocess') as pool:
        pending=pool.submit(_prepare_cpu,loaded,images,settings)
        while True:
            inputs=pending.result()
            try:next_images,next_labels=next(it)
            except StopIteration:next_images=None
            if next_images is not None:pending=pool.submit(_prepare_cpu,loaded,next_images,settings)
            yield _move(inputs,loaded.device),labels
            if next_images is None:break
            labels=next_labels
