"""Complete independent convex-head namespace; correct SHA dictionary identity."""
import ast,inspect
import caltech20_rich_alpha40_samehead as m
import audit_caltech20_rich_robust_head as original
tree=ast.parse(inspect.getsource(original.audit));changed=0
class Identity(ast.NodeTransformer):
    def visit_Compare(self,node):
        global changed
        if ast.unparse(node)=="m.s.digest(opt['sources']) == m.s.digest(meta['sources'])":changed+=1;return ast.parse("opt['sources'] == meta['sources']",mode='eval').body
        return self.generic_visit(node)
tree=Identity().visit(tree);assert changed==1;namespace=dict(vars(original));namespace['m']=m;exec(compile(ast.fix_missing_locations(tree),__file__+':same_independent_convex_science','exec'),namespace);audit=namespace['audit']
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--family',choices=list(m.PHASES),required=True);p.add_argument('--depth',type=int,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args();m.setup(a.family)
    try:audit(a.depth,a.phase)
    except BaseException as e:
        out=m.OUT/f'L{a.depth}_rho03';mark=out/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
