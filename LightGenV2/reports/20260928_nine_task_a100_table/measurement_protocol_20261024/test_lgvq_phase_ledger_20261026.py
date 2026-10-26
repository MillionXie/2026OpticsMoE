import unittest
from lgvq_phase_ledger_20261026 import longest_path_partition, event_phases, PASSES,software_only_submissions


class PhaseTests(unittest.TestCase):
    def test_only_exact_software_submission_nodes_excluded_not_queue_or_electronics(self):
        raw=dict(records=[dict(field=5,events=[dict(kind=k) for k in
            ('frontend','electronics','optical_simulation','detector_ROI','electronic_residual')])])
        self.assertEqual(software_only_submissions(raw),{'f5.H2Dsubmit','f5.submit2','f5.submit3','f5.submit4'})
        self.assertNotIn('f5.g2.queue',software_only_submissions(raw))
    def test_overlap_not_summed_into_wall(self):
        nodes=[dict(id='a',kind='frontend',parents=[],duration_ms=8),
               dict(id='b',kind='electronics',parents=[],duration_ms=6),
               dict(id='c',kind='electronics',parents=['a','b'],duration_ms=2)]
        out=longest_path_partition(nodes,{n['id']:n['duration_ms'] for n in nodes},{'b':'oeo1','c':'decoder'},1)
        self.assertEqual(out['wall_ms16'],10)
        self.assertEqual(sum(out['critical_path_contribution_ms16'].values()),10)
        self.assertNotIn('oeo1',out['critical_path_contribution_ms16'])

    def test_six_ordered_physical_passes_and_detail_required(self):
        events=[dict(stream='body',kind='optical_simulation',label=p) for p in PASSES]
        raw=dict(records=[dict(field=0,events=events)])
        self.assertEqual(event_phases(raw)['f0.g5'],'optics6')
        raw['detailed_electronics_phases']=True
        with self.assertRaisesRegex(ValueError,'merger'):event_phases(raw)
        events.extend([dict(stream='body',kind='electronics',label='frame_merger'),
                       dict(stream='body',kind='electronics',label='readout')])
        self.assertEqual(event_phases(raw)['f0.g7'],'decoder')


if __name__=='__main__':unittest.main()
