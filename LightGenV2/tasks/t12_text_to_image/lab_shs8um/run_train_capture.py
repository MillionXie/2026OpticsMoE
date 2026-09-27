"""Version-pinned TRAIN adapter; unchanged physical forward and layer ordering."""
from pathlib import Path
from . import run_layerwise

def main():
    code=Path(run_layerwise.__file__).read_text(encoding='utf8')
    changes={
        "choices=('test','val'),default='test'":"choices=('train',),default='train'",
        "p.add_argument('--max-samples',type=int)":"p.add_argument('--max-samples',type=int);p.add_argument('--selection',type=Path,required=True)",
        "assert len(dataset)==2304":"assert len(dataset)==20736",
        "selected_indices=list(range(len(dataset))) if a.max_samples is None else torch.linspace(0,len(dataset)-1,min(a.max_samples,len(dataset))).long().tolist()":"selection=json.loads(a.selection.read_text(encoding='utf8')); assert selection['split']=='train' and selection['test_product_overlap']==0 and selection['test_source_hash_overlap']==0; selected_indices=selection['indices']; assert len(selected_indices)==len(set(selected_indices)); assert a.max_samples is None; write(a.output/'selection.json',selection)",
        "'test_index'":"'train_index'",
        "f'test_":"f'train_",
    }
    for old,new in changes.items():
        assert old in code,('Pinned capture adapter source changed',old)
        code=code.replace(old,new)
    exec(compile(code,run_layerwise.__file__,'exec'),dict(__name__='__main__',__package__=run_layerwise.__package__))

if __name__=='__main__':main()
