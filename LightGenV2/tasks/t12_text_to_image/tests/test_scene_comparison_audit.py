import unittest
from LightGenV2.tasks.t12_text_to_image.scene_comparison_audit import paired_intervals,route_summary


class PairedSceneAudit(unittest.TestCase):
    def rows(self):
        return [dict(sample_id=str(i),reference_id='pillow-B123-'+str(i%2),category='pillow',condition_id=str(i),seed=1042,prompt='same',
            input_sha256='ref',initial_latent_sha256='noise',CLIP_T=.3,DINOv2_I=.4,CLIP_I=.8,ImageReward_compact=-2.) for i in range(6)]

    def test_views_and_seeds_share_a_product_and_negative_reward_is_difference(self):
        a=self.rows();b=[dict(r,CLIP_T=r['CLIP_T']*.9,ImageReward_compact=-1.5) for r in a]
        result=paired_intervals(a,b,100);self.assertEqual(result['products'],1)
        self.assertAlmostEqual(result['metrics']['CLIP_T']['paired_relative_change'],-.1)
        self.assertEqual(result['metrics']['ImageReward_compact']['paired_absolute_change'],.5)
        with self.assertRaises(ValueError):paired_intervals(a,[dict(r,seed=4) for r in b],100)

    def test_static_hard_routes_are_not_mistaken_for_balanced_routing(self):
        diag=dict(alpha=[.65,.65],router_calls_per_forward=1,physical_active_roi=478,propagation_canvas=518,expert_tile=224,top_k=2,selected_mask=[[1,0,1,0]])
        rows=[dict(actual_raw_optical_routing=dict(vision=diag)) for _ in range(8)]
        d=route_summary(rows)['vision'];self.assertFalse(d['observed_routing_varies_across_requests']);self.assertEqual(d['expert_selection_fractions'],[1.,0.,1.,0.])
        with self.assertRaises(ValueError):route_summary([dict(actual_raw_optical_routing=dict(vision=dict(diag,alpha=[.5,.65])))])


if __name__=='__main__':unittest.main()
