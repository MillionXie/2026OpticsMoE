import numpy as np
import caltech20_rich_covariance_correction4 as m
def independent_statistics(x,y):
 centers=np.zeros((20,256));counts=np.zeros(20,int)
 for label in range(20):centers[label]=np.sum(x[y==label],axis=0)/sum(y==label);counts[label]=sum(y==label)
 cov=np.zeros((256,256))
 for label in range(20):d=x[y==label]-centers[label];cov+=d.T@d
 return centers,cov/(len(y)-20),counts
def run():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_TRAIN_covariance_linear_solve');q,pt,pred=m.incumbent();x=pred['train']['features'].astype(float);y=pred['train']['labels'];a,b,c=independent_statistics(x,y);aa,bb,cc=m.statistics(x,y);assert np.allclose(a,aa,atol=1e-12) and np.allclose(b,bb,atol=1e-12) and np.array_equal(c,cc);assert pred['train']['ids'].tolist()==m.base.manifest()['split_ids']['train'];w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();records=[]
 for beta in m.config()['shrinkage']:
  w,bi,scale=m.correction(x,y,w0,b0,beta);matrix=(1-beta)*b+beta*np.trace(b)/256*np.eye(256);error=float(np.max(abs((w/scale)@matrix-a)));assert error<1e-8 and np.linalg.eigvalsh(matrix).min()>0;records.append(dict(beta=beta,normal_equation_maxabs=error,TRAIN_scale=scale))
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=records,optical_models=0,optical_forward=0,optimizer=0,TEST_used_for_covariance=False),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
