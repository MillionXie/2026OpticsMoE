"""CPU-only scheduling contracts; no model or CUDA imports."""
import unittest
import lgvq_dynamic_cpu_20261101 as policy
import lgvq_compute_scope_20261029 as scope


class DynamicContract(unittest.TestCase):
    def fixture(self,fields=3):
        phases=['emb_text','emb_conv3d','emb_position','adapter_visual_pool','adapter_visual',
                'merger','adapter_language','decoder']+[f'oeo{i}' for i in range(1,7)]
        report=dict(fields=fields,raw_phases=[dict(field=f,phase=p,charged_compute_s=.001)
                    for f in range(fields) for p in phases])
        rows=[dict(field=f,slot=s,worker_thread=999,resize_cpu_s=.010,
                   normalization_cpu_s=.020,patch_cpu_s=.010,sample_id=f'{f}-{s}')
              for f in range(fields) for s in range(16)]
        tokens=[dict(thread_cpu_s=.001) for _ in range(fields)]
        return report,rows,tokens

    def test_no_cpu_backend_backpressure(self):
        nodes=policy.projected_nodes(*self.fixture(),3)
        cpu=[n for n in nodes if n['resource'].startswith('cpu_dynamic_')]
        self.assertEqual(len({n['resource'] for n in cpu}),16)
        self.assertFalse(any('decoder' in p for n in cpu for p in n['parents']))
        self.assertTrue(any('f0.decoder' in n['parents'] for n in nodes if n['id']=='f2.tokenizer'))

    def test_continuous_fifo_and_no_fixed_threads(self):
        schedule=scope.compute_schedule(policy.projected_nodes(*self.fixture(),3))
        cpu=[n for n in schedule['nodes'] if n['resource'].startswith('cpu_dynamic_')]
        for worker in range(16):
            jobs=[n for n in cpu if n['resource']==f'cpu_dynamic_{worker}']
            self.assertAlmostEqual(jobs[0]['start_s'],0)
            for a,b in zip(jobs,jobs[1:]):self.assertAlmostEqual(a['end_s'],b['start_s'])
        self.assertAlmostEqual(max(n['end_s'] for n in cpu),.120)

    def test_1886_fixed_optics_not_FFT(self):
        nodes=policy.projected_nodes(*self.fixture(),3)
        optics=[n for n in nodes if n['resource']=='optical']
        self.assertEqual(len(optics),18)
        self.assertTrue(all(n['duration_s']==1/1886+.000300+.0000307 for n in optics))


if __name__=='__main__':unittest.main()
