"""Output-only safety boundary for historical T10 report transformations."""
from pathlib import Path


def prepare_output(source, output, required):
    source, output = Path(source).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Existing reports are protected: ' + str(output))
    target = output.resolve()
    if target == source or source in target.parents:
        raise ValueError('Derived output must be outside the original evidence package')
    missing = [str(source / path) for path in required if not (source / path).is_file()]
    if missing:
        raise FileNotFoundError('Missing original evidence: ' + '; '.join(missing))
    target.mkdir(parents=True, exist_ok=False)
    return target
