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


def restrict_training_to(model,prefixes):
    parameters=dict(model.named_parameters())
    selected=set()
    for prefix in prefixes:
        matches={name for name in parameters if name==prefix or name.startswith(prefix+'.')}
        if not matches:
            raise ValueError('Unknown trainable parameter prefix: '+prefix)
        selected.update(matches)
    if not selected:
        raise ValueError('Select at least one trainable parameter')
    for name,value in parameters.items():
        value.requires_grad_(name in selected)
    return sorted(set(parameters)-selected)
