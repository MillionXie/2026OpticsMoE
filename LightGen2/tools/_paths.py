"""Resolve this standalone task for script and package entry points."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
TOOLS=Path(__file__).resolve().parent

if __package__ in (None,'','tools'):
    for directory in (ROOT,TOOLS):
        if str(directory) not in sys.path:
            sys.path.insert(0,str(directory))
