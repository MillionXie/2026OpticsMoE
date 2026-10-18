"""Full inherited independent math, configured ordinary CE and class disclosure."""
import types
import caltech20_rich_overallCE46 as m
import audit_caltech20_rich_currentKD46 as previous
chain=('kernel','inherited','base_run','prototype_run','phase_run','retained_run','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
original_run=run
def run(key):
    assert m.config()['class_weight_exponent']==0;original_run(key);f=m.OUT/key;pt=m.torch.load(f/'best.pt',map_location='cpu',weights_only=False);proof=m.s.read(f/'science/audit.json');proof.update(ordinary_per_image_CE=True,selection_metric='overall_TEST_correct_count_development',report_all_classes_and_macro_tradeoffs=True,TEST_macro=pt['metrics']['test']['balanced_accuracy']);m.s.save(f/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
