"""Local independent metric, source, split, checkpoint, and budget verification."""
import hashlib,json,sys,tarfile
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'.codex_plot_deps'))
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'completed';OUT.mkdir(exist_ok=True)
archive=OUT/'completed_results.tar.gz';meta=json.loads((OUT/'ARCHIVE_SHA256.json').read_text(encoding='utf-8'))
assert archive.stat().st_size==meta['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==meta['sha256']
dest=OUT/'archive';dest.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
    seen=set()
    for item in tar:
        assert item.isfile() and item.name not in seen;seen.add(item.name)
        target=(dest/item.name).resolve();assert target.is_relative_to(dest.resolve())
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(item).read())
manifest=json.loads((dest/'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest,name
root=dest/'moe_root';d2=dest/'d2nn_root'
source=json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
assert source==json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
for name,digest in source.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
split=json.loads((ROOT/'SPLIT.json').read_text(encoding='utf-8'))['records']
signature=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert signature(split)==manifest['split_sha256']
assert json.loads((root/'DATA_CHECKS.json').read_text(encoding='utf-8'))['passed']
checks=json.loads((root/'runs/preflight.json').read_text(encoding='utf-8'));assert checks['passed'] and checks['source_sha256']==signature(source)
data_manifest=json.loads((dest/'dataset/IMAGE_MANIFEST.json').read_text(encoding='utf-8'));assert len(data_manifest)==len(split)
assert {x['path'] for x in split}==set(data_manifest)
assert not json.loads((root/'DATA_CHECKS.json').read_text(encoding='utf-8'))['cross_split_or_label_duplicates']
seal=json.loads((root/'runs/SELECTION_SEAL.json').read_text(encoding='utf-8'));performance=json.loads((root/'results/PERFORMANCE.json').read_text(encoding='utf-8'))
rows=[]
for model,entry in performance['models'].items():
    assert entry['same_fixed_checkpoint_both_domains']
    assert seal['models'][model]['sha256']==entry['checkpoint']['sha256']==manifest['selected_checkpoint_sha256'][model]
    row={'model':model}
    for domain in ('A','B'):
        indices=np.array([i for i,r in enumerate(split) if r['domain']==domain and r['split']=='test'])
        expected=np.array([split[i]['label'] for i in indices])
        with np.load(root/'results'/model/f'{domain}_test.npz',allow_pickle=False) as z:
            assert np.array_equal(z['indices'],indices) and np.array_equal(z['labels'],expected)
            logits=z['logits'];assert logits.shape==(len(indices),10) and np.isfinite(logits).all()
            p=logits.argmax(1);assert np.array_equal(p,z['predictions'])
            if model=='moe':
                assert (z['route_weights']>0).all();assert np.allclose((z['route_weights']**2).sum(1),1,atol=1e-5)
            cm=np.bincount(10*expected+p,minlength=100).reshape(10,10)
            acc=float((p==expected).mean());f1=float((2*np.diag(cm)/np.maximum(1,cm.sum(0)+cm.sum(1))).mean())
            report=entry['tests'][domain]
            assert acc==report['accuracy'] and f1==report['macro_f1'] and cm.tolist()==report['confusion_matrix']
            row[domain+'_accuracy']=acc;row[domain+'_macro_f1']=f1
    row['mean_accuracy']=(row['A_accuracy']+row['B_accuracy'])/2;assert row['mean_accuracy']==entry['mean_accuracy'];rows.append(row)
stages=[('shared',40,250),('expert_A',15,125),('expert_B',15,125),('router',15,250)]
for stage,epochs,steps in stages:
    m=root/'runs/moe'/stage;d=d2/'runs/AB'/stage
    assert len(list(m.glob('epoch_*.json')))==len(list(d.glob('epoch_*.json')))==epochs
    for epoch in range(1,epochs+1):
        a=json.loads((m/f'epoch_{epoch:03d}.json').read_text(encoding='utf-8'));b=json.loads((d/f'epoch_{epoch:03d}.json').read_text(encoding='utf-8'))
        assert a['steps']==b['steps']==steps and a['epoch']==b['epoch']==epoch
        for key in ('train_samples','domain_presentations','sample_order_and_augmentation_sha256'):assert a[key]==b[key]
        assert a['frozen_unchanged'] and b['frozen_unchanged'] and b['matched_moe_data']
for model,steps,domain in [('A_only',125,'A'),('B_only',125,'B')]:
    folder=d2/'runs'/model/model;assert len(list(folder.glob('epoch_*.json')))==70
    for epoch in range(1,71):
        h=json.loads((folder/f'epoch_{epoch:03d}.json').read_text(encoding='utf-8'));assert h['steps']==steps and h['epoch']==epoch
        assert set(h['validation'])=={domain,'mean','mean_ce'}
        assert h['domain_presentations']['B' if domain=='A' else 'A']==0
assert seal['models']['moe']['domain_presentations']==seal['models']['AB']['domain_presentations']=={'A':525000,'B':525000}
assert seal['models']['moe']['total_steps']==seal['models']['AB']['total_steps']==17500

plan=json.loads((root/'PLAN.json').read_text(encoding='utf-8'))
assert performance['seed']==42 and not performance['test_based_selection']
for partition in ('train','validation','test'):
    sets=[{r['pair_id'] for r in split if r['domain']==d and r['split']==partition} for d in ('A','B')]
    assert sets[0]==sets[1]
assert sum(sum(c.values()) for c in plan['counts'].values())==len(split)
checks=json.loads((root/'DATA_CHECKS.json').read_text(encoding='utf-8'))
assert checks['paired_split_consistent'] and checks['spatial_groups_disjoint'] and checks['cross_split_minimum_center_distance_m']>=3000
names={'moe':'光路由 MoE A+B','AB':'D2NN A+B','A_only':'D2NN A-only','B_only':'D2NN B-only'}
lines=['# EuroSAT 光学/SAR 双域测试性能','','A 为光学 RGB，B 为 SAR。每个模型用一个固定检查点测试两个域；F1 为 Macro-F1。','','| 模型 | A 准确率 | B 准确率 | A F1 | B F1 | 两域平均 | 最差域 |','|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    row['worst_accuracy']=min(row['A_accuracy'],row['B_accuracy'])
    assert row['worst_accuracy']==performance['models'][row['model']]['worst_accuracy']
    lines.append('| '+names[row['model']]+' | '+' | '.join(f"{row[k]*100:.2f}%" for k in ('A_accuracy','B_accuracy','A_macro_f1','B_macro_f1','mean_accuracy','worst_accuracy'))+' |')
lines += ['', '实际划分：'+json.dumps(plan['counts'],ensure_ascii=False), '', '四模型共享输入编码与地理划分；MoE与D2NN AB每轮采样及增强哈希完全一致，各17,500步、A/B各525,000次样本呈现。两者光学参数量、损失和教师计算量不同。单seed42，不据测试结果调参。', '', '所有预测、指标、混淆矩阵、数据配对、源代码和选定检查点哈希已独立核对。恢复相位和均匀/隔离路由诊断须另行从NPZ复算解释。']
(OUT/'EuroSAT性能报告.md').write_text('\n'.join(lines),encoding='utf-8')
verification=dict(passed=True,archive_sha256=meta['sha256'],source_sha256=signature(source),split_sha256=signature(split),verified_rows=rows,counts=plan['counts'])
(OUT/'LOCAL_VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False,indent=2))
