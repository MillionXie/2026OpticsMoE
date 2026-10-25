import unittest,numpy as np
from .project_stage_teacher import held_TRAIN


class StageProjectionTests(unittest.TestCase):
    def test_TRAIN_holdout_preserves_pairs_and_tasks(self):
        ids=np.concatenate([np.arange(300),np.arange(300),np.arange(1000)]);tasks=np.array(['eurosat']*300+['clevr']*300+['speech_binary']*1000);y=np.concatenate([np.arange(300)%10,np.arange(300)%2,np.arange(1000)%2]);ix=held_TRAIN(ids,y,tasks)
        self.assertEqual(len(ix),320);self.assertEqual(len(set(ix.tolist())),320)
        for n in ('clevr','speech_binary'):
            values=ids[ix][tasks[ix]==n];self.assertTrue(all((v^1) in values for v in values))
        np.testing.assert_array_equal(ix,held_TRAIN(ids,y,tasks))


if __name__=='__main__':unittest.main()
