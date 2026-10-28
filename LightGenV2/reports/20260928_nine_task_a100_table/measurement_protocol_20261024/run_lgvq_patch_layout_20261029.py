"""Explicit distinct layout-only candidate for the approved compute protocol."""
from pathlib import Path
import run_lgvq_compute_scope_20261029 as runtime
import lgvq_compute_scope_20261029 as scope
import lgvq_patch_layout_20261029 as layout

ENTRY=Path(__file__).resolve();NATIVE_SOURCES=runtime.sources

def sources():
    return {**NATIVE_SOURCES(),**{p.relative_to(runtime.REPO).as_posix():runtime.sha(p)
        for p in (ENTRY,Path(runtime.__file__),Path(layout.__file__))}}

if __name__=='__main__':
    # Each fresh process explicitly binds this different implementation. The
    # previously published baseline source and failed roots remain unchanged.
    runtime.ENTRY=ENTRY;runtime.sources=sources;scope.patch_video=layout.patch_video
    runtime.main()
