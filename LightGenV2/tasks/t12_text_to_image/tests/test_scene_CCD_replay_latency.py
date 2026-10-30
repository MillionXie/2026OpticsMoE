import unittest
from ..scene_CCD_replay_latency import wait_optical_period,request_matches,PERIOD_SECONDS


class ReplayTiming(unittest.TestCase):
    def test_interrupted_sleep_cannot_undercount_the_optical_period(self):
        now=[0.];calls=[]
        def clock():return now[0]
        def sleep(t):calls.append(t);now[0]+=t/2 if len(calls)==1 else t
        actual=wait_optical_period(clock=clock,sleep=sleep)
        self.assertEqual(len(calls),2);self.assertAlmostEqual(actual,PERIOD_SECONDS*1000)
        self.assertAlmostEqual(6*actual,5.1655361611877)
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):wait_optical_period(value)
    def test_frames_are_bound_to_both_RGB_and_complete_text(self):
        row=dict(sample_id='lamp-1',prompt='Generate a lamp. Include a bookshelf.',reference_sha256='abc')
        self.assertTrue(request_matches(row,dict(row)))
        for key in row:
            changed=dict(row);changed[key]+='changed'
            self.assertFalse(request_matches(row,changed))


if __name__=='__main__':unittest.main()
