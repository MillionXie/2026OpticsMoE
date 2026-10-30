import copy
import unittest
import torch
from LightGenV2.tasks.t12_text_to_image.scene_trajectory_coarsen import coarsen


class CoarsenedTeacher(unittest.TestCase):
    def saved(self):
        sigmas=[8.,6.,4.,2.,0.];points=[torch.tensor([i*.5,i*-.25]) for i in range(5)]
        return dict(student_steps=4,teacher_steps=30,timesteps=[90.,70.,50.,30.],sigmas=sigmas,records=[dict(split='train',input_key='a',interval=i,x=points[i],target_epsilon=(points[i+1]-points[i])/(sigmas[i+1]-sigmas[i]),sigma=sigmas[i],timestep=90.-20*i) for i in range(4)])
    def test_merged_intervals_reach_actual_teacher_endpoints(self):
        original=self.saved();new,edges,error=coarsen(original,2)
        self.assertEqual(edges,[0,2,4]);self.assertEqual(new['sigmas'],[8.,4.,0.]);self.assertEqual(error,0.)
        for i,row in enumerate(new['records']):
            endpoint=row['x']+row['target_epsilon']*(new['sigmas'][i+1]-row['sigma'])
            expected=original['records'][2]['x'] if i==0 else torch.tensor([2.,-1.])
            torch.testing.assert_close(endpoint,expected)
        self.assertEqual(original['student_steps'],4)
    def test_rejects_missing_nonTRAIN_or_nonmatching_endpoints(self):
        for mode in ['missing','dev','corrupt','duplicate']:
            saved=copy.deepcopy(self.saved())
            if mode=='missing':saved['records'].pop()
            if mode=='dev':saved['records'][0]['split']='dev'
            if mode=='corrupt':saved['records'][1]['x']+=1
            if mode=='duplicate':saved['records'].append(copy.deepcopy(saved['records'][0]))
            with self.assertRaises(ValueError):coarsen(saved,2)


if __name__=='__main__':unittest.main()
