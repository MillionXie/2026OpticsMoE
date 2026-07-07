# Quality detector regions

The final optical detector contains three fixed, non-overlapping regions assigned to `high_quality`, `medium_quality`, and `low_quality`. Training combines electronic-readout cross entropy, region-distribution cross entropy, and a detector-energy concentration penalty. The region distribution is also supplied as three semantic features to the small electronic readout.
