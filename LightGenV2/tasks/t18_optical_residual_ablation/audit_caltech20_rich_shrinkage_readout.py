"""Standalone independent saved CPU head audit entry, no old stage re-entry."""
import caltech20_rich_shrinkage_readout as m
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args();m.audit(a.phase)
