import types
import unittest
import torch
from memory import inference_policy,place_inference_model,release_transients,cast_fp32

class MemoryTests(unittest.TestCase):
    def test_cpu_policy(self):
        dev,low,fp32=inference_policy('cpu')
        self.assertEqual(str(dev),'cpu'); self.assertFalse(low); self.assertTrue(fp32)
    def test_token_lookup_unchanged(self):
        model=torch.nn.Module(); model.language=torch.nn.Module()
        model.language.embed_tokens=torch.nn.Embedding(20,8)
        ids=torch.tensor([[1,2,5]])
        before=model.language.embed_tokens(ids).detach().clone()
        weights=model.language.embed_tokens.weight.detach().clone()
        place_inference_model(model,model.language,torch.device('cpu'),cpu_token_table=True)
        torch.testing.assert_close(model.language.embed_tokens(ids),before,rtol=0,atol=0)
        torch.testing.assert_close(model.language.embed_tokens.weight,weights,rtol=0,atol=0)
    def test_cache_release_keeps_parameters(self):
        model=torch.nn.Linear(4,4); original=model.weight.detach().clone()
        model.last_latent_groups=[torch.ones(1,4)]; model._stage1_latent=torch.ones(1,4)
        model.last_routing={'probabilities':torch.ones(1,4)}
        b=types.SimpleNamespace(loaded=types.SimpleNamespace(model=model),replacement=types.SimpleNamespace(last_language_hidden=torch.ones(1)),device=torch.device('cpu'))
        release_transients(b)
        self.assertEqual(model.last_latent_groups,[]); self.assertIsNone(model._stage1_latent)
        self.assertEqual(model.last_routing,{})
        torch.testing.assert_close(model.weight,original,rtol=0,atol=0)
    def test_selected_rows_fp32_match_full_fp32_table(self):
        model=torch.nn.Module(); model.language=torch.nn.Module()
        model.language.embed_tokens=torch.nn.Embedding(20,8).to(torch.bfloat16)
        model.head=torch.nn.Linear(8,4).to(torch.bfloat16)
        ids=torch.tensor([[1,2,5]])
        expected=model.language.embed_tokens(ids).float().detach().clone()
        cast_fp32(model,model.language,preserve_cpu_table=True)
        self.assertEqual(model.language.embed_tokens.weight.dtype,torch.bfloat16)
        self.assertEqual(model.head.weight.dtype,torch.float32)
        place_inference_model(model,model.language,torch.device('cpu'),cpu_token_table=True,embedding_output_dtype=torch.float32)
        actual=model.language.embed_tokens(ids)
        self.assertEqual(actual.dtype,torch.float32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

if __name__=='__main__': unittest.main()
