"""All revised pairs and all matched DEV requests, with explicit factual QA.

No global image/text proxy is presented as instruction success. Uncertain and
missing facts remain visible, alongside the complete public prompt and pixels.
"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil
from .scene_instruction_data import PROFILE,condition,sha,write


def signature(row):return tuple(row[k]for k in ['sample_id','reference_sha256','prompt','seed','category','condition_id'])


def checked_run(path,requests):
    report=json.loads((path/'report.json').read_text(encoding='utf8'));images=json.loads((path/'images.json').read_text(encoding='utf8'))
    if not report['complete'] or report['prompt_profile']!=PROFILE or report['independent_TEST_opened'] or report['resolution']!=512:
        raise ValueError('Completed new DEV protocol only')
    if len(images)!=48 or {signature(r)for r in images}!={signature(r)for r in requests}:raise ValueError('Full same48 reference/prompt/seed cohort required')
    indexed={r['sample_id']:r for r in images}
    for row in images:
        if sha(path/row['generated_PNG'])!=row['generated_sha256']:raise ValueError('Changed generated pixels')
    return report,indexed


def build(data,models,output,review=None):
    if output.exists():raise ValueError('New output directory required')
    report=json.loads((data/'report.json').read_text(encoding='utf8'));pairs=json.loads((data/'pairs.json').read_text(encoding='utf8'))
    requests=json.loads((data/'inference_inputs.json').read_text(encoding='utf8'))
    if report['prompt_profile']!=PROFILE or not report['complete'] or report['TEST_opened'] or sha(data/'pairs.json')!=report['pairs_sha256'] or len(requests)!=48:
        raise ValueError('Complete SHA-bound revised dataset required')
    checked=[(spec,*checked_run(Path(spec['run']),requests))for spec in models]
    if not checked:raise ValueError('At least one actual baseline output required')
    first=checked[0][2]
    for _,_,images in checked[1:]:
        if any(images[k]['initial_latent_sha256']!=first[k]['initial_latent_sha256']for k in first):
            raise ValueError('Matched actual initial noise required')
    if review is not None:
        for spec,_,images in checked:
            annotations=review['models'][spec['key']]
            if set(annotations)!=set(images):raise ValueError('Full manual review denominator required')
            if any(annotations[k]['generated_sha256']!=images[k]['generated_sha256']for k in images):raise ValueError('Review does not bind displayed output pixels')
    output.mkdir(parents=True);assets=output/'images';assets.mkdir();escape=html.escape;parts=[]
    def copy(src,name,expected):
        if sha(src)!=expected:raise ValueError('Data pixels changed')
        dst=assets/name
        if not dst.exists():shutil.copyfile(src,dst)
        return 'images/'+name
    counts=Counter(r['split']for r in pairs);products=len({r['source_product_id']for r in pairs if r['split']=='train'})
    parts.append('<!doctype html><meta charset="utf-8"><title>数据修订与指令遵从验证</title><style>body{font:16px Arial,sans-serif;margin:24px;background:#fafafa;color:#222}article{background:white;border:1px solid #ddd;border-radius:8px;margin:18px 0;padding:16px}.grid{display:flex;gap:12px;overflow-x:auto}figure{margin:0;min-width:240px;width:300px}img{width:100%;height:auto}figcaption{font-weight:bold;margin:8px 0}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}code{overflow-wrap:anywhere}.warn{background:#fff0d2;padding:14px}.facts{font-size:14px;color:#444}</style>')
    parts.append('<h1>数据修订与指令遵从验证</h1><p class="warn">本页完整保留所有配对与48条验证请求。新数据的目标图是训练示例，不能当作新模型输出；旧图不换提示词重新宣称成功。全局CLIP／IR等代理均值不能证明指令要求已经出现。本页不是独立TEST500或新LG结果。</p>')
    parts.append(f'<p>训练配对{counts["train"]}；DEV噪声监督{counts["dev"]}；TRAIN商品{products}。目标512×512，白底参考仍输入Qwen256。生成验证保留原16个DEV视图，各3条新提示，共48请求。</p>')
    parts.append('<h2>模型身份</h2><table><tr><th>模型</th><th>采样步数</th><th>权重SHA</th><th>训练状态</th></tr>')
    for spec,r,_ in checked:
        new=r.get('expanded_DEV_protocol',{}).get('model_trained_on_revised_dataset',False)
        parts.append('<tr><td>'+escape(spec['label'])+'</td><td>'+str(r['sampling_steps'])+'</td><td><code>'+escape(r['checkpoint_sha256'])+'</code></td><td>'+('新事实数据重训；仍需逐项验收'if new else'旧权重对新文字的先导，不是重训结果')+'</td></tr>')
    parts.append('</table><h2>全部48条指令验证</h2>')
    for i,row in enumerate(requests):
        ref=copy(data/row['reference'],f'dev_{i:02d}_reference.png',row['reference_sha256']);parts.append('<article><h3>'+escape(row['sample_id'])+'</h3><p>'+escape(row['prompt'])+'</p><div class="grid"><figure><figcaption>白底输入</figcaption><img src="'+ref+'"></figure>')
        for spec,_,images in checked:
            r=images[row['sample_id']];im=copy(Path(spec['run'])/r['generated_PNG'],f'dev_{i:02d}_{spec["key"]}.png',r['generated_sha256'])
            status='逐项检查待完成'
            if review is not None:
                a=review['models'][spec['key']][row['sample_id']];status=a['status']+'：'+a['note']
            parts.append('<figure><figcaption>'+escape(spec['label'])+'</figcaption><img src="'+im+'" loading="lazy"><p class="facts">'+escape(status)+'</p></figure>')
        parts.append('</div></article>')
    parts.append('<h2>全部修订配对：训练目标，非模型新输出</h2>')
    for i,row in enumerate(pairs):
        ref=copy(data/row['reference'],f'pair_{i:03d}_reference.png',row['reference_sha256']);target=copy(data/row['target'],f'pair_{i:03d}_target.png',row['target_sha256'])
        parts.append('<article><h3>'+escape(row['split']+' / '+row['sample_id'])+'</h3><p>'+escape(row['prompt'])+'</p><p class="facts">要求核对：'+escape('；'.join(row['required_visible_facts']))+'</p><div class="grid"><figure><figcaption>白底参考</figcaption><img src="'+ref+'" loading="lazy"></figure><figure><figcaption>已核对的合成训练示例</figcaption><img src="'+target+'" loading="lazy"></figure></div><details><summary>原文字（保留审计，不再用于此配对）</summary><p>'+escape(row['original_prompt'])+'</p></details></article>')
    (output/'index.html').write_text(''.join(parts),encoding='utf8')
    if review is not None:write(output/'visible_fact_review.json',review)
    write(output/'delivery_manifest.json',dict(complete=True,source_dataset=str(data),pairs_sha256=report['pairs_sha256'],paired_images=len(pairs),all48_requests_shown=True,
        models=[dict(label=s['label'],report_sha256=sha(Path(s['run'])/'report.json'),checkpoint_sha256=r['checkpoint_sha256'])for s,r,_ in checked],
        instruction_success_certified=False,independent_TEST_opened=False,review_present=review is not None))
    write(output/'file_manifest.json',{p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))for p in output.rglob('*')if p.is_file()and p.name!='file_manifest.json'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--specification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.specification.read_text(encoding='utf8'));review=json.loads(Path(spec['review']).read_text(encoding='utf8'))if spec.get('review')else None
    build(Path(spec['data']),spec['models'],a.output,review)


if __name__=='__main__':main()
