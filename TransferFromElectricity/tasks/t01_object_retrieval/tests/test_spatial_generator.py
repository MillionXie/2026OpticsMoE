import unittest
from types import SimpleNamespace

import torch
from torch import nn

from TransferFromElectricity.tasks.t01_object_retrieval.models.spatial_generator import (
    QVLoRA, SpatialDecoder, SpatialGenerator, select_references, unmerge_tokens)
from TransferFromElectricity.tasks.t01_object_retrieval.models.injection import GlobalInjection
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import PhaseLayer
from TransferFromElectricity.tasks.t01_object_retrieval.protocol import zero_optical_phases


class SpatialTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(42)

    def test_block_order_inverts_to_raster_and_keeps_gradient(self):
        raster=torch.arange(4*6*3,dtype=torch.float32).reshape(4,6,3).requires_grad_()
        blocked=raster.reshape(2,2,3,2,3).permute(0,2,1,3,4).reshape(24,3)
        result=unmerge_tokens(blocked,torch.tensor([[1,4,6]]),2)
        torch.testing.assert_close(result[0],raster.permute(2,0,1),rtol=0,atol=0)
        result[0,1,2,3].backward()
        self.assertEqual(float(raster.grad[2,3,1]),1)
        self.assertEqual(float(raster.grad.sum()),1)

    def test_fused_lora_changes_qv_but_preserves_k(self):
        base=nn.Linear(6,18);adapter=QVLoRA(base,2);x=torch.randn(3,6)
        reference=base(x).detach();torch.testing.assert_close(adapter(x),reference,rtol=0,atol=0)
        adapter(x).square().sum().backward()
        self.assertGreater(float(adapter.lora_b.grad.norm()),0)
        self.assertIsNone(base.weight.grad)
        with torch.no_grad():adapter.lora_b.add_(.1)
        output=adapter(x)
        torch.testing.assert_close(output[:,6:12],reference[:,6:12],rtol=0,atol=0)
        self.assertGreater(float((output-reference).abs().max()),0)

    def test_dense_decoder_has_neighbor_dependencies_and_exact_geometry(self):
        decoder=SpatialDecoder();x=torch.randn(1,512,14,14,requires_grad=True)
        output=decoder(x);self.assertEqual(tuple(output.shape),(2,4,224,224))
        output[0,0,112,112].backward()
        spatial_support=x.grad.abs().sum(1)[0]>0
        self.assertGreater(int(spatial_support.sum()),1)
        self.assertEqual(tuple(SpatialDecoder(2,True)(x.detach()).shape),(2,478,478))

    def test_global_injection_keeps_autograd_and_materializes_exactly(self):
        planes=[PhaseLayer(478),PhaseLayer(478)];injection=GlobalInjection(planes)
        generated=torch.randn(2,478,478,requires_grad=True)
        injection.bind(generated)
        before=torch.stack([p.phase() for p in planes])
        before.square().mean().backward()
        self.assertGreater(float(generated.grad.norm()),0)
        injection.materialize()
        torch.testing.assert_close(before,torch.stack([p.phase() for p in planes]),rtol=0,atol=0)
        self.assertTrue(all(not p.raw_phase.requires_grad for p in planes))

    def test_global_stays_zero_until_unlock_without_phase_jump(self):
        generator=SpatialGenerator.__new__(SpatialGenerator);nn.Module.__init__(generator)
        generator.generates_global=True;generator.global_enabled=False
        generator.decoder=SpatialDecoder();generator.global_decoder=SpatialDecoder(2,True)
        features=torch.randn(1,512,14,14)
        generator.spatial_features=lambda:features
        generator.register_buffer('initial_reference',generator.decoder(features).detach())
        generator.register_buffer('global_reference',torch.zeros(2,478,478))
        generator.register_buffer('global_reference_ready',torch.tensor(False))
        self.assertEqual(float(generator().abs().max()),0)
        features.add_(.4);generator()
        self.assertEqual(float(generator.current_global.abs().max()),0)
        generator.set_global_enabled(True);generator()
        self.assertEqual(float(generator.current_global.abs().max()),0)
        reference=generator.global_reference.clone()
        with torch.no_grad():generator.global_decoder.head.bias.add_(.1)
        generator.set_global_enabled(True);generator()
        self.assertGreater(float(generator.current_global.abs().max()),.09)
        torch.testing.assert_close(reference,generator.global_reference,rtol=0,atol=0)

    def test_references_are_fixed_training_only_and_class_balanced(self):
        training=[SimpleNamespace(sku_index=c,sample_id=f'train:{c}:{i}') for c in range(3) for i in range(10)]
        a=select_references(training,42);b=select_references(list(reversed(training)),42)
        self.assertEqual([r.sample_id for r in a],[r.sample_id for r in b])
        self.assertEqual([r.sku_index for r in a],[0,1,2])

    def test_zero_initialization_includes_distinct_router_parameter(self):
        class Router(nn.Module):
            def __init__(self):super().__init__();self.raw_router_phase=nn.Parameter(torch.ones(8,8))
            def phase(self):return 2*torch.pi*self.raw_router_phase.sigmoid()
        module=nn.ModuleDict({'expert':PhaseLayer(8,init='normal'),'global':PhaseLayer(12,init='normal'),'router':Router()})
        audit=zero_optical_phases(module)
        self.assertEqual(len(audit),3)
        self.assertIn('router.raw_router_phase',audit)
        self.assertTrue(all(row['raw_max_abs']==0 for row in audit.values()))


if __name__=='__main__':unittest.main()
