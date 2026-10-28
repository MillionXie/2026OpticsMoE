import unittest
from .safeguarded_refine import armijo_accept,direction,pool_indices,attach_test_split


class Safeguards(unittest.TestCase):
    def test_strict_decrease_and_finite(self):
        self.assertTrue(armijo_accept(1.,.99,-.2,1.))
        self.assertFalse(armijo_accept(1.,1.,-.2,1.))
        self.assertFalse(armijo_accept(1.,.99,.2,1.))
        self.assertFalse(armijo_accept(1.,float('nan'),-.2,1.))
        self.assertFalse(armijo_accept(1.,.9999995,-1.,1.))

    def test_curved_quadratic_descent(self):
        import numpy as np
        x=np.array([1.,-2.]);h=np.array([2.,5.]);scales=np.array([1.,.5]);memory=[]
        for _ in range(8):
            grad=h*x;p,info=direction(grad,memory,scales,.25)
            self.assertLess(info['directional_derivative'],0);self.assertLessEqual(np.linalg.norm(p),.250001)
            before=float((h*x*x).sum()/2);alpha=1.
            for _ in range(4):
                after=float((h*(x+alpha*p)**2).sum()/2)
                if armijo_accept(before,after,float(grad@p),alpha):break
                alpha*=.5
            self.assertTrue(armijo_accept(before,after,float(grad@p),alpha))
            nxt=x+alpha*p;s=nxt-x;y=h*nxt-grad
            memory=(memory+[(s,y,1/float(s@y))])[-3:];x=nxt
        self.assertLess(float((h*x*x).sum()/2),11.)

    def test_uniform_TRAIN_and_all_TEST(self):
        import numpy as np
        labels=np.concatenate([np.repeat(np.arange(8),437),np.repeat(np.arange(8),94)])
        indices=pool_indices(labels,27)
        self.assertEqual(len(indices),1504);self.assertEqual(set(indices[indices>=3496]),set(range(3496,4248)))
        self.assertTrue(np.array_equal(np.bincount(labels[indices[indices<3496]],minlength=8),np.repeat(94,8)))
        self.assertTrue(np.array_equal(indices,pool_indices(labels,27)))
        self.assertFalse(np.array_equal(indices,pool_indices(labels,37)))

    def test_explicit_deferred_TEST_split(self):
        import numpy as np
        x=np.arange(4248)[:,None];y=np.arange(4248)%8;ids=np.arange(4248)
        splits=dict(train=(x,y,ids),original_train=(x[:3496],y[:3496],ids[:3496]))
        self.assertNotIn('test',splits);attach_test_split(splits)
        self.assertTrue(np.array_equal(splits['test'][2],np.arange(3496,4248)))
        self.assertEqual(len(splits['test'][1]),752)


if __name__=='__main__':unittest.main()
