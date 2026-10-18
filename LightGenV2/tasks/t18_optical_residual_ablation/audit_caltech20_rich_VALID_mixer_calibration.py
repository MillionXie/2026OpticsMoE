"""Own exclusive saved science, without another optical/test forward."""
import caltech20_rich_VALID_mixer_calibration as m
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:m.audit(a.phase)
    except BaseException as e:
        mark=m.OUT/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
