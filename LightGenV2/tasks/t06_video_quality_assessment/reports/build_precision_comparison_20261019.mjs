import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(path.join(process.cwd(),'.codex_tmp/artifact-runtime.cjs'));
const {FileBlob,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const repo=process.cwd(),out=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013');
const root=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_rental_precision_20261019/presentation');
const data=JSON.parse(await fs.readFile(path.join(root,'precision_comparison.json'),'utf8'));
const template=path.join(out,'LGVQ_busy50_near_saturation_plot_data.xlsx');
const original=await fs.readFile(template);
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const archive=path.join(root,'prior_report_archive');await fs.mkdir(archive,{recursive:true});
try{await fs.writeFile(path.join(archive,path.basename(template)),original,{flag:'wx'});}catch(e){if(e.code!=='EEXIST')throw e;}
const colors={ours:'#7A3E9D',qwen:'#0072B2',glm:'#9B567C',clip:'#009E73',yolo:'#E69F00',full:'#D55E00',kimi:'#4A70A0'};
const names={ours:'LightGen-II',qwen:'Qwen3-VL-2B',glm:'GLM-4.6V-Flash',clip:'CLIP ViT-B/32',yolo:'YOLO11s',full:'DeepSeek-VL2',kimi:'Kimi-VL-A3B'};
const receipts=[];
for(const precision of ['native','fp32']){
 const book=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(archive,path.basename(template))));
 const sheet=book.worksheets.getItem('Plot data'),inputs=book.worksheets.getItem('Measurements');
 const before=await book.render({sheetName:'Plot data',range:'A1:G19',scale:1,format:'png'});
 await fs.writeFile(path.join(repo,'.codex_tmp',precision+'_precision_before.png'),new Uint8Array(await before.arrayBuffer()));
 const rows=[data.groups[precision][0]];
 for(const baseline of ['qwen','glm','clip','yolo','full','kimi'])for(const condition of ['Approximately 50%','Near-saturation reference']){
  rows.push(data.groups[precision].find(x=>x.baseline===baseline&&x.condition===condition)??{
   baseline,model:names[baseline],condition,precision,plotted:false,missing:true,
   notes:baseline==='full'&&precision==='fp32'?'FP32 weights require109.9GB, exceed80GB VRAM; no offload or quantization.':'No complete validated measurement available.'});
 }
 sheet.getRange('A2').values=[[precision==='native'?'Deployed precision: approximately50% and near saturation':'FP32 execution: approximately50% and near saturation']];
 sheet.getRange('A4').values=[['First200 distinct TEST videos; first computation block to GPU output. Pacing waits excluded. Utilization independently measured.']];
 const keys=['model','baseline','condition','batch','srcc','utilization_percent','time_ms','energy_j','board_w','host_proxy_w',
  'cost_wall_sum_ms','cost_video_count','active_samples','excluded_wait_16_ms','parameters_billion','platform','gpu_uuid',
  'cost_source','utilization_source','checkpoint_sha256','precision','repeat1','repeat2','repeat3','status','notes'];
 inputs.getRange('A6:Z6').values=[keys];
 inputs.getRange('A21').values=[['Historical deployed-precision audit inputs below; not FP32 results']];
 for(const[i,x]of rows.entries()){
  const r=7+i,repeat=x.repeats??x.util_repeats??[];
  const values={...x,repeat1:repeat[0],repeat2:repeat[1],repeat3:repeat[2],status:x.missing?'unavailable':x.target50_failed?'outside target':'measured'};
  inputs.getRange(`A${r}:Z${r}`).values=[keys.map(k=>values[k]??null)];
  const available=!x.missing&&x.plotted!==false;
  sheet.getRange(`A${r}:L${r}`).values=[[x.model,x.condition,x.batch??null,null,null,null,null,null,null,x.parameters_billion??null,x.platform??null,values.status]];
  if(available){
   sheet.getRange(`D${r}:I${r}`).formulas=[[`=Measurements!E${r}`,x.baseline==='ours'?'="n.a."':`=Measurements!F${r}/100`,
    `=Measurements!G${r}`,`=Measurements!H${r}`,`=F${r}/$F$7`,`=G${r}/$G$7`]];
  }else sheet.getRange(`D${r}:I${r}`).values=[['n.a.','n.a.','n.a.','n.a.','n.a.','n.a.']];
  const color=colors[x.baseline];sheet.getRange(`A${r}:L${r}`).format.fill=i%2?'#F4F7F9':'#FFFFFF';
  sheet.getRange(`A${r}`).format.font={name:'Arial',size:10,color,bold:true};
  if(!available)sheet.getRange(`L${r}`).format.font={name:'Arial',size:10,color:'#9C3D10'};
 }
 sheet.getRange('A22:L38').clear({applyTo:'contents'});
 const notes=[['Cost boundary','First native computation block input to synchronized GPU MOS output. No model warmup before the first cost batch.'],
  ['Energy estimate','(338.2W retained host proxy + same-cost-run sampled active board W) × pure inference seconds.'],
  ['Independent utilization','First-B resident four-frame inputs; sequential queue8, three complete5s warmup/30s measurement repeats, all samples pooled.'],
  ['Pacing','Real inference followed by idle intervals. Calibration frozen before formal repeats. Waiting retained separately, excluded from formal pure costs.'],
  ['Accuracy','558 unique TEST videos independently evaluated for every new precision/setting. No retraining or TEST-selected precision.'],
  ['Precision',precision==='fp32'?'All floating parameters/input and computationFP32, autocast/TF32 disabled. Expanded deployment weights do not recover lost values; official eager attention where required.':'Original deployed BF16/FP16 backbones andFP32 quality heads; four accepted original-platform points reused.'],
  ['Platforms','Original four models: PCIe40GB/250W. Kimi andfullDeepSeek: SXM4-80GB/400W. Explicitly different deployments, not a common-hardware causal ranking.'],
  ['Missing points','Missing or failed measurements remain unavailable; no estimated point or interpolated performance is plotted. QwenB1 finalized data unchanged.']];
 for(const[i,[label,text]]of notes.entries()){
  const r=22+i*2;sheet.getRange('A'+r).values=[[label]];sheet.getRange(`B${r}:L${r}`).unmerge();sheet.getRange(`B${r}:L${r}`).merge();sheet.getRange('B'+r).values=[[text]];
  sheet.getRange(`A${r}:L${r}`).format={font:{name:'Arial',size:10,color:'#53687A'},wrapText:true,rowHeight:32};
 }
 inputs.getRange('A6:Z6').format={fill:'#214D68',font:{name:'Arial',bold:true,color:'#FFFFFF',size:10},rowHeight:42,wrapText:true};
 inputs.getRange('A7:Z19').format.font={name:'Arial',size:10,color:'#243447'};
 inputs.getRange('A7:Z19').format.rowHeight=26;
 inputs.getRange('E7:O19').setNumberFormat('0.000000');inputs.getRange('V7:X19').setNumberFormat('0.000000');
 inputs.getRange('R6:T19').format.columnWidth=65;inputs.getRange('Z6:Z19').format.columnWidth=90;
 inputs.getRange('N6:Q19').format.columnWidth=28;
 sheet.getRange('D7:D19').setNumberFormat('0.000000');sheet.getRange('E7:E19').setNumberFormat('0.0%');
 sheet.getRange('F7:G19').setNumberFormat('#,##0.000');sheet.getRange('H7:I19').setNumberFormat('0.00"×"');
 book.recalculate();
 const verify=b=>{
  const s=b.worksheets.getItem('Plot data');
  for(const[i,x]of rows.entries()){
   const r=7+i;
   if(x.missing||x.plotted===false){if(typeof s.getRange('F'+r).values[0][0]==='number')throw Error('Missing point became numeric');continue;}
   for(const[c,k]of [['D','srcc'],['F','time_ms'],['G','energy_j']])if(Math.abs(s.getRange(c+r).values[0][0]-x[k])>1e-8)throw Error('Precision value mismatch');
   if(x.baseline!=='ours'&&Math.abs(s.getRange('E'+r).values[0][0]-x.utilization_percent/100)>1e-12)throw Error('Utilization mixed');
  }
 };
 verify(book);
 const active=rows.findIndex(x=>x.baseline!=='ours'&&!x.missing&&x.plotted!==false);
 if(active>=0){const r=active+7,oldValue=inputs.getRange('G'+r).values[0][0];inputs.getRange('G'+r).values=[[oldValue*1.01]];book.recalculate();
  if(Math.abs(sheet.getRange('F'+r).values[0][0]-oldValue*1.01)>1e-8)throw Error('Formula recalculation failed');
  inputs.getRange('G'+r).values=[[oldValue]];book.recalculate();verify(book);}
 const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:15},maxChars:1000})).ndjson;
 if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
 for(const[sheetName,range]of [['Plot data','A1:G19'],['Measurements','A6:N12']]){
  const image=await book.render({sheetName,range,scale:1.2,format:'png'});
  await fs.writeFile(path.join(repo,'.codex_tmp',precision+'_precision_'+sheetName.replaceAll(' ','_')+'.png'),new Uint8Array(await image.arrayBuffer()));
 }
 const target=path.join(out,precision==='native'?'LGVQ_busy50_near_saturation_plot_data.xlsx':'LGVQ_fp32_busy50_near_saturation_plot_data.xlsx');
 await(await SpreadsheetFile.exportXlsx(book)).save(target);verify(await SpreadsheetFile.importXlsx(await FileBlob.load(target)));
 receipts.push({precision,file:path.relative(repo,target).replaceAll('\\','/'),sha256:digest(await fs.readFile(target)),formula_errors:0,recalculation:true,reimport:true,rows:13,available:rows.filter(x=>!x.missing&&x.plotted!==false).length});
}
await fs.writeFile(path.join(root,'workbook_receipt.json'),JSON.stringify({source_sha256:digest(await fs.readFile(path.join(root,'precision_comparison.json'))),workbooks:receipts},null,2)+'\n');
console.log(JSON.stringify({workbooks:receipts}));
