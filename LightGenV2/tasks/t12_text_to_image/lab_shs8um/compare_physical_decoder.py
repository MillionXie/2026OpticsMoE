"""All TEST rows, matched baseline/tuned images, plus disclosed figure shortlist."""
import argparse,csv,hashlib,json,shutil,time,zipfile
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--tuned',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    deadline=time.monotonic()+10800
    while not (a.tuned/'report.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Tuned TEST replay did not complete')
        time.sleep(15)
    assert not a.output.exists();a.output.mkdir(parents=True)
    baseline=json.loads((a.baseline/'sample_metrics.json').read_text());tuned=json.loads((a.tuned/'sample_metrics.json').read_text());assert len(baseline)==len(tuned)==2304
    lookup={r['test_index']:r for r in baseline};rows=[]
    for t in tuned:
        i=t['test_index'];b=lookup[i];assert b['prompt']==t['prompt'] and b['source_sample_id']==t['source_id']
        r=dict(test_index=i,sample_id=f'test_{i:05d}',source_id=t['source_id'],prompt=t['prompt'],category=t['category'],mode=t['mode'])
        for label,source,key in [('reference',a.baseline,b['reference_image']),('target',a.baseline,b['target_image']),('physical_baseline',a.baseline,b['physical_image']),('simulation_baseline',a.baseline,b['simulation_image']),('physical_tuned',a.tuned,t['physical_tuned_image']),('simulation_tuned',a.tuned,t['simulation_tuned_image'])]:
            original=source/key;destination=a.output/'images'/label/f'test_{i:05d}.png';destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,destination);r[label+'_image']=str(destination.relative_to(a.output));r[label+'_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
        for domain in ('physical','simulation'):
            for metric in ('mse_0_1','mae_0_1','psnr_db','ssim'):
                r[domain+'_baseline_'+metric]=b[domain+'_'+metric];r[domain+'_tuned_'+metric]=t[domain+'_'+metric]
        r['physical_psnr_improvement_db']=r['physical_tuned_psnr_db']-r['physical_baseline_psnr_db'];r['physical_ssim_improvement']=r['physical_tuned_ssim']-r['physical_baseline_ssim'];rows.append(r)
    def save(name,data):
        (a.output/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
        with (a.output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save('all_test_comparison',rows)
    groups={}
    for row in rows:groups.setdefault((row['category'],row['mode']),[]).append(row)
    picks=[]
    for members in groups.values():
        chosen={r['test_index']:r for key in ('physical_tuned_ssim','physical_psnr_improvement_db') for r in sorted(members,key=lambda r:r[key],reverse=True)[:8]}
        picks.extend(chosen.values())
    save('figure_shortlist',picks)
    (a.output/'README.md').write_text('# Full matched TEST comparison\n\n2304 rows, six native256 lossless PNGs per row. Join by test_index/sample_id. Reference is input; target is expected edit; baseline/tuned are matched same-CCD predictions. Prompt and source identity retained. MSE/MAE RGB[0,1], PSNR range1, SSIM Gaussian11 sigma1.5 valid windows.\n\nFigure shortlist selects up to8 best SSIM and8 largest PSNR improvements in each category/mode for presentation only, not weight selection or population metrics. All2304 remain in all_test_comparison; aggregate every row, not this shortlist. No fake upscale. FP32 predictions, TRAIN/TEST rawCCD and best/last retained in original runs.\n',encoding='utf8')
    totals={k:sum(r[k] for r in rows)/len(rows) for k,v in rows[0].items() if isinstance(v,float)}
    (a.output/'report.json').write_text(json.dumps(dict(status='complete',sample_count=2304,shortlist_count=len(picks),metrics=totals,tuned_report=json.loads((a.tuned/'report.json').read_text())),indent=2),encoding='utf8')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for file in a.output.rglob('*'):
            if file.is_file():z.write(file,file.relative_to(a.output))
if __name__=='__main__':main()
