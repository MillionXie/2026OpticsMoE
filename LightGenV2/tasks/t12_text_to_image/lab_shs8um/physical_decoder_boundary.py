"""Only post-optical decoder parameters may change during physical adaptation."""
import hashlib

PREFIXES=('editor.up3.','editor.up2.','editor.up1.','editor.to_delta.','editor.source_gate.')

def is_decoder(name):
    return name.startswith(PREFIXES)

def freeze_for_adaptation(model):
    model.eval().requires_grad_(False)
    selected=[]
    for name,parameter in model.named_parameters():
        if is_decoder(name):
            parameter.requires_grad_(True)
            selected.append((name,parameter))
    assert selected and any(name.startswith('editor.to_delta.') for name,_ in selected)
    assert not any('bottleneck' in name or 'optical' in name for name,_ in selected)
    return selected

def protected_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if not is_decoder(name):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()
