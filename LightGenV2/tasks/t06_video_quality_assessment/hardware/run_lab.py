"""Portable handoff entry; inference uses only pinned Qwen-front caches."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'runtime'))
from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import main
if __name__=='__main__':main()
