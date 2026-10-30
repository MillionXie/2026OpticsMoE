import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from ..scene_acceleration_delivery import checked_timing,timing_html
from ..white_reference_scene_probe import digest


def fixture(root,mode='software'):
    model=dict(role='light_lg',label='LG6',report=dict(checkpoint_sha256='fixed',sampling_steps=6),images={})
    rounds=[]
    for repeat in range(3):
        folder=root/str(repeat);folder.mkdir();rounds.append(str(folder));rows=[]
        for index in range(8):
            image=folder/(str(index)+'.png');image.write_bytes(str(index).encode())
            row=dict(sample_id=str(index),input_sha256='reference',prompt='Generate a scene. Include a plant.',seed=index,initial_latent_sha256='noise',category='lamp',condition_id=index,reference_id='view'+str(index//4),generated_PNG=image.name,generated_sha256=digest(image))
            rows.append(row);model['images'][str(index)]=row
        (folder/'images.json').write_text(json.dumps(rows));(folder/'execution.json').write_text(json.dumps(dict(pid=100+repeat)))
        (folder/'report.json').write_text(json.dumps(dict(complete=True,scope='DEV_software_latency',checkpoint_sha256='fixed',sampling_steps=6)))
        (folder/'software_latency.json').write_text(json.dumps(dict(n=8,requests_ms=[700.]*8,first_request_ms=700.,batch_size=1,extra_warmup=False,model_load_excluded=True,task_LoRA_fused=True,output_PNG_identity_vs_completed_quality_run=True,contains_GPU_FFT_optical_simulation=mode=='software',optical_wait_emulation=mode!='software')))
        (folder/'CCD_replay_latency.json').write_text(json.dumps(dict(complete=True,FFT_propagation_calls=0,physical_CCD_or_SDK_used=False,electronic_residual_and_fusion_retained=True,CPU_payload_conversion_and_device_copies_retained=True,fixed_optical_ms=5.1655361611877)))
    return dict(mode=mode,rounds=rounds),model


class AccelerationDeliveryTest(unittest.TestCase):
    def test_all24_requests_and_two_reference_views_are_reported(self):
        with TemporaryDirectory()as temp:
            spec,model=fixture(Path(temp));value=checked_timing(spec,model)
            self.assertEqual((value['n'],value['reference_views'],value['mean_ms']),(24,2,700.))

    def test_profiler_wall_and_changed_checkpoint_are_rejected(self):
        with TemporaryDirectory()as temp:
            spec,model=fixture(Path(temp));path=Path(spec['rounds'][0])/'software_latency.json';r=json.loads(path.read_text());r['contains_profiler_overhead']=True;path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'uninstrumented'):checked_timing(spec,model)
            r['contains_profiler_overhead']=False;path.write_text(json.dumps(r));model['report']['checkpoint_sha256']='other'
            with self.assertRaisesRegex(ValueError,'displayed'):checked_timing(spec,model)

    def test_wait_emulation_never_becomes_a_physical_measurement(self):
        with TemporaryDirectory()as temp:
            spec,model=fixture(Path(temp),'ideal_CCD_fixed_wait');r=checked_timing(spec,model)
            self.assertFalse(r['physical_measurement']);self.assertTrue(r['unknown_extra_device_transport_not_assumed_zero'])
            text=timing_html([dict(role='qwen',label='E30',mode='software',mean_ms=2800.),r])
            self.assertEqual(r['extra_hardware_H_budget_for4x_ms'],0.)
            self.assertIn('不是实际CCD',text)

    def test_same_pid_or_changed_quality_pixels_are_rejected(self):
        with TemporaryDirectory()as temp:
            spec,model=fixture(Path(temp));path=Path(spec['rounds'][1])/'execution.json';path.write_text(json.dumps(dict(pid=100)))
            with self.assertRaisesRegex(ValueError,'Independent'):checked_timing(spec,model)
            path.write_text(json.dumps(dict(pid=101)));model['images']['0']=dict(model['images']['0'],generated_sha256='wrong')
            with self.assertRaisesRegex(ValueError,'displayed'):checked_timing(spec,model)


if __name__=='__main__':unittest.main()
