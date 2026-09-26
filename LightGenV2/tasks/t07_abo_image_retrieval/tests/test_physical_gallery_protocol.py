"""Lightweight tests for the measured-to-measured retrieval evaluator.

Extract only the pure evaluator; do not import vendor SDKs or open hardware.
"""
import ast
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def evaluator():
    path=Path(__file__).parents[1]/'lab_dvp8um/full_query_flow.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    namespace={'F':F,'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['evaluate']


def test_gallery_uses_supplied_measured_vectors_and_query_order():
    rows=[{'sample_id':'qa','product_id':'a','split':'query'},
          {'sample_id':'qb','product_id':'b','split':'query'}]
    gallery=[{'sample_id':'gb','product_id':'b','split':'train'},
             {'sample_id':'ga','product_id':'a','split':'train'}]
    protocol={'rows':gallery+rows}
    query=torch.tensor([[1.,0.],[0.,1.]])
    physical={'ids':['gb','ga'],'vectors':torch.tensor([[0.,1.],[1.,0.]])}
    predictions,metrics=evaluator()(query,rows,protocol,physical)
    assert metrics['recall_at_1']==1.0
    assert [r['top1_sample_id'] for r in predictions]==['ga','gb']
    mixed={'ids':['gb','ga'],'vectors':torch.tensor([[1.,0.],[0.,1.]])}
    _,metrics=evaluator()(query,rows,protocol,mixed)
    assert metrics['recall_at_1']==0.0
