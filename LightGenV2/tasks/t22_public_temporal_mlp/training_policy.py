"""Training-only parameter freezing with explicit, checked module names."""
from __future__ import annotations


def freeze_parameters(model,prefixes):
    frozen=[]
    for prefix in prefixes:
        matches=[(name,value) for name,value in model.named_parameters()
                 if name==prefix or name.startswith(prefix+'.')]
        if not matches:
            raise ValueError('Unknown frozen parameter prefix: '+prefix)
        for name,value in matches:
            value.requires_grad_(False)
            frozen.append(name)
    return sorted(set(frozen))
