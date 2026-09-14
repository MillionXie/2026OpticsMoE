"""Standalone entry. No hardware access; uses only packaged measured caches."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from LightGenV2.tasks.t06_video_quality_assessment.adapt_measured_readout import main
if __name__=='__main__':main()
