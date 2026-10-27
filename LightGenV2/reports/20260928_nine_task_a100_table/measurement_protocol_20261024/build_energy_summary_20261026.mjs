// Final qualified JSON only. Run the spreadsheet skill marker immediately
// before the first authoring execution; pure layout tests do not create files.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

export const stages=[
  ['CPU_image_preparation_excluded_ms','图像准备，排除'],
  ['H2D_excluded_ms','原始搬运，排除'],
  ['CPU_tokenizer_retained_ms','CPU文字分词'],
  ['entry_dispatch_retained_ms','模型入口调度'],
  ['full_GPU_model_to_score_host_ms','完整GPU模型至评分'],
  ['intermediate_CPU_output_retained_ms','中间CPU返回'],
  ['intervideo_dispatch_retained_ms','视频间调度'],
  ['final_CPU_output_outside_primary_ms','末条CPU返回，窗外'],
  ['GPU_model_entry_to_first_TF_host_ms','GPU前端至首网络节点'],
  ['first_TF_to_score_host_ms','首网络节点至评分'],
];
const labels={qwen:'Qwen',glm:'GLM',clip:'CLIP',yolo:'YOLO',kimi:'Kimi',deepseek:'DeepSeek完整'};
const platform=p=>p==='A10040GB_PCIE_250W'?'40GB PCIe / 250W':'80GB SXM4 / 400W';
export const col=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const groupKey=p=>`${p.platform}/${p.model}/${p.cap}`;
const close=(a,b)=>{if(typeof a!=='number'||typeof b!=='number'||!Number.isFinite(a+b)||Math.abs(a-b)>Math.max(1e-5,Math.abs(b)*1e-9))throw Error(`Numeric mismatch ${a} ${b}`);};
const sha=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');

export function layout(data){
  if(data.status!=='final_qualified_inputs'||data.lab_complete_points!==24||!data.LG_async_measured||!data.LG||data.missing_lab_points.length)throw Error('Final qualified baseline and LG inputs required');
  const expected=new Set(['qwen','glm','clip','yolo'].flatMap(m=>[50,100].flatMap(c=>[1,2,3].map(r=>`${m}/${c}/${r}`))));
  const lab=data.rounds.filter(p=>p.platform==='A10040GB_PCIE_250W');
  if(lab.length!==24||new Set(lab.map(p=>`${p.model}/${p.cap}/${p.repeat}`)).size!==24||lab.some(p=>!expected.has(`${p.model}/${p.cap}/${p.repeat}`)))throw Error('Incorrect 24-point identities');
  const groups=data.groups.map((g,i)=>({...g,summaryRow:7+i}));
  const rounds=[];let row=2;
  for(const g of groups){
    const points=data.rounds.filter(p=>groupKey(p)===groupKey(g)).sort((a,b)=>a.repeat-b.repeat);
    if(points.length!==g.rounds||!points.length)throw Error('Group count mismatch');
    if(new Set(points.map(p=>p.repeat)).size!==points.length)throw Error('Duplicate original rounds');
    g.lo=row;g.hi=row+points.length-1;
    for(const p of points)rounds.push({...p,row:row++});
  }
  if(rounds.length!==data.rounds.length)throw Error('Unmapped original points');
  return {groups,rounds,lgInputStart:row+3};
}

export async function build(input,out,phasePath=null,refreshUnsealed=false){
  const data=JSON.parse(await fs.readFile(input,'utf8'));const plan=layout(data);
  const deployed=Boolean(data.LG_deployed_profile);
  if(deployed&&(data.LG_rounds?.length!==3||data.LG.video_calls!==6144))throw Error('Three complete new LG rounds required');
  const rental=new Set(data.rounds.filter(p=>p.platform==='A10080GB_SXM4_400W').map(p=>`${p.model}/${p.cap}/${p.repeat}`));
  if(data.rounds.length!==36||rental.size!==12||!['kimi','deepseek'].every(m=>[50,100].every(c=>[1,2,3].every(r=>rental.has(`${m}/${c}/${r}`)))))throw Error('Unified delivery requires all 12 rental identities plus 24 lab points');
  const phase=phasePath?JSON.parse(await fs.readFile(phasePath,'utf8')):null;
  if(phase&&(phase.status!=='passed'||path.resolve(phase.source_leaf)!==path.resolve(data.LG.path)||!phase.sum_only_additive_wall_column))throw Error('Qualified matching LG phase ledger required');
  let existing=false;try{await fs.access(out);existing=true;}catch(e){if(e.code!=='ENOENT')throw e;}
  if(existing){
    if(!refreshUnsealed)throw Error('Preserve existing output directory');
    try{await fs.access(path.join(out,'saved_workbook_audit.json'));throw Error('Never refresh sealed/audited output');}catch(e){if(e.code!=='ENOENT')throw e;}
    const previous=JSON.parse(await fs.readFile(path.join(out,'builder_checks.json'),'utf8'));
    if(previous.input_summary_sha256!==await sha(input)||previous.workbook_sha256!==await sha(path.join(out,'Temporal_LG与全部baseline统一时间能量功率_20261027.xlsx')))throw Error('Preserve unknown existing changes');
  }
  for(const r of data.raw_file_index){
    if((await fs.stat(r.local_path)).size!==r.bytes||await sha(r.local_path)!==r.sha256)throw Error('Input evidence changed: '+r.local_path);
  }
  const {Workbook,SpreadsheetFile}=await import('@oai/artifact-tool');
  const w=Workbook.create();
  const summary=w.worksheets.add('结果与阶段'),rounds=w.worksheets.add('轮次原值'),index=w.worksheets.add('原始文件索引');
  const physical=phase?w.worksheets.add('物理阶段'):null;
  const base=(s,rows,cols)=>{s.showGridLines=false;s.getRangeByIndexes(0,0,rows,cols).format={font:{name:'Arial',size:11,color:'#182B3A'},rowHeight:25,verticalAlignment:'center'};};
  base(summary,80,14);base(rounds,plan.lgInputStart+20,32);base(index,data.raw_file_index.length+1,6);
  summary.tabColor='#243B53';
  const put=(s,r,rows)=>s.getRangeByIndexes(r-1,0,rows.length,rows[0].length).values=rows;
  const header=(s,r,n)=>{s.getRangeByIndexes(r-1,0,1,n).format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:56};};
  const title=(s,r,text,n)=>{put(s,r,[[text]]);s.getRangeByIndexes(r-1,0,1,n).format={font:{size:15,bold:true},rowHeight:30,borders:{bottom:{style:'thin',color:'#8FA4B5'}}};};
  const formula=(s,cell,value)=>s.getRange(cell).formulas=[[value]];
  if(physical){
    base(physical,65,12);physical.tabColor='#526D82';
    physical.getRange('A1:A1').format.columnWidth=44;physical.getRange('B1:D1').format.columnWidth=22;
    physical.getRange('E1:E1').format.columnWidth=39;physical.getRange('F1:L1').format.columnWidth=21;
    title(physical,2,'LG：物理阶段与并行墙钟分解',5);
    put(physical,4,[[deployed?'每16视频，三轮合并。首场保留，仅电子图构建单列；只有墙钟贡献列可求和。':'每16视频。节点跨度可重叠，调度含观测等候；只有墙钟贡献列可求和。首场和真实建图保留。']]);
    put(physical,5,[['光电口径剔除FFT/ROI/电残差/H2D及其CPU提交，再按原依赖重排。原因不明的观测排队保留。']]);
    put(physical,7,[['阶段','节点跨度/ms','总墙钟贡献/ms','占墙钟比例','记录范围']]);header(physical,7,5);
    for(const [i,p] of phase.rows.entries()){
      const r=8+i;put(physical,r,[[deployed&&p.key==='startup'?'电子图构建（部署项）':p.block,p.executed_work_ms16,p.additive_wall_contribution_ms16,null,
        p.key==='frontend'?'embedding、位置、adapter合并打点':p.key==='scheduling'?'含CPU返回、提交与必要同步':
        p.key==='startup'?(deployed?'电子图构建单列，原软件仍含该成本':'真实首次建图准备，不摊销删去'):p.key==='excluded'?'光电口径置零，软件仍真实执行':
        p.key.startsWith('optics')?'固定光周期，非硬件计时':p.key.startsWith('oeo')?'原路由/读出、融合和下一段编码':
        ['merger','decoder'].includes(p.key)&&!phase.merger_decoder_independently_measured?'未单独打点，归入相邻OEO':'原事件边界']]);
      formula(physical,'D'+r,`=C${r}/$C$30`);
    }
    if(phase.rows.length>22)throw Error('Physical phase table exceeds reserved layout');
    put(physical,30,[['合计（仅可加墙钟列）',null,null,1,'工作跨度不求和']]);formula(physical,'C30',`=SUM(C8:C${7+phase.rows.length})`);
    put(physical,31,[['独立主时间',null,phase.fixed_optics.wall_ms16]]);
    put(physical,32,[['墙钟闭合差',null,null]]);formula(physical,'C32','=C30-C31');
    put(physical,34,[['Qwen MPS50对照/ms',data.groups.find(g=>g.model==='qwen'&&g.cap===50).time16_ms]]);
    put(physical,35,[['100倍门槛/ms',null]]);formula(physical,'B35','=B34/100');
    put(physical,36,[['光电投影时间比',null]]);formula(physical,'B36','=B34/C30');
    put(physical,37,[['距100倍门槛/ms',null]]);formula(physical,'B37','=C30-B35');
    put(physical,38,[[deployed?'本三轮含建图v2/ms':'本轮旧口径/ms',deployed?phase.previous_v2_time16_ms:phase.legacy_fixed_optics_time16_ms]]);
    put(physical,39,[[deployed?'电子图构建关键路径/ms':'旧轮同新口径/ms',deployed?phase.critical_path_reduction_from_graph_build_ms16:phase.reference_round?.same_scope_time16_ms??'不适用']]);
    physical.getRange('B8:C37').setNumberFormat('0.000000');physical.getRange('D8:D30').setNumberFormat('0.0%');
    physical.getRange('A8:A29').format.rowHeight=30;physical.getRange('E8:E29').format.rowHeight=30;
    title(physical,42,'baseline：串行主时间按对应阶段归纳',12);
    put(physical,44,[['模型','MPS/%','平台','文字tokenizer/ms','GPU前端/ms','主干与decoder/ms','调度含CPU返回/ms','主时间合计/ms','独立主时间/ms','闭合差/ms','末条CPU返回/ms','至末CPU返回/ms']]);header(physical,44,12);
    for(const [i,g] of plan.groups.entries()){
      const r=45+i,stage=g.stage_equivalent16_ms;
      const front=stage.GPU_model_entry_to_first_TF_host_ms,body=stage.first_TF_to_score_host_ms;
      const source=(key)=>stage[key]??0;
      put(physical,r,[[labels[g.model],g.cap,g.platform==='A10040GB_PCIE_250W'?'40GB/250W':'80GB/400W',source('CPU_tokenizer_retained_ms'),
        front??'未独立记录',body??stage.full_GPU_model_to_score_host_ms,
        source('entry_dispatch_retained_ms')+source('intermediate_CPU_output_retained_ms')+source('intervideo_dispatch_retained_ms'),
        null,g.time16_ms,null,source('final_CPU_output_outside_primary_ms'),null]]);
      formula(physical,'H'+r,`=SUM(D${r}:G${r})`);formula(physical,'J'+r,`=H${r}-I${r}`);formula(physical,'L'+r,`=H${r}+K${r}`);
    }
    physical.getRange(`D45:L${44+plan.groups.length}`).setNumberFormat('0.000');physical.getRange('B38:B39').setNumberFormat('0.000000');
    physical.getRange(`B45:C${44+plan.groups.length}`).format.horizontalAlignment='center';physical.getRange(`D45:L${44+plan.groups.length}`).format.horizontalAlignment='right';
    put(physical,57,[['租机无首网络节点独立记录，GPU前端并入主干与decoder。baseline没有光传播阶段。']]);
    put(physical,58,[['原baseline未逐embedding/adapter/merger/decoder打点，不事后估分。中间CPU返回并入调度；末次返回单列并给含它的总时间。']]);
    put(physical,60,[['LG阶段来源SHA256',await sha(phasePath)]]);
  }
  const fileByPath=new Map(data.raw_file_index.map(r=>[path.resolve(r.local_path),r]));
  const digest=(p,name)=>{const r=fileByPath.get(path.resolve(p.path,name));if(!r)throw Error('Missing source index: '+name);return r.sha256;};
  const sourceCols=['原cost绝对路径','原cost SHA256','本地audit SHA256','输出NPZ SHA256','首网络节点是否独立记录'];
  put(rounds,1,[['平台','模型','MPS百分比','轮次','视频调用数','保留累计时间/s','同区间代理能量/J','共同有效tokens','自身有效tokens',
    '每16视频/ms','每16视频/J','共同tokens/s','自身tokens/s',...stages.map(([,label])=>label+'，累计ms'),...sourceCols,
    '硬件功率上限/W','同保留区间板卡积分/J','同保留区间板卡平均/W','固定主机代理/W']]);
  header(rounds,1,32);rounds.freezePanes.freezeRows(1);rounds.freezePanes.freezeColumns(4);
  rounds.getRangeByIndexes(1,0,plan.rounds.length,32).format.rowHeight=26;
  for(const p of plan.rounds){
    const native=p.native_effective_tokens;
    put(rounds,p.row,[[platform(p.platform),labels[p.model],p.cap,p.repeat,p.calls,p.elapsed_s,p.energy_j,p.common_effective_tokens,native,
      null,null,null,null,...stages.map(([key])=>p.stage_totals_ms[key]??null),path.join(p.path,'cost_report.json'),
      digest(p,'cost_report.json'),digest(p,'local_independent_energy_audit01.json'),digest(p,'outputs.npz'),p.first_TF_node_recorded?'是':'未独立记录',
      p.hardware_power_limit_w,p.measured_board_energy_j,null,p.host_proxy_w]]);
    formula(rounds,'AE'+p.row,`=AD${p.row}/F${p.row}`);
    for(const [c,f] of [['J',`=F${p.row}*1000/E${p.row}*16`],['K',`=G${p.row}/E${p.row}*16`],['L',`=H${p.row}/F${p.row}`]])formula(rounds,c+p.row,f);
    if(native===null)rounds.getRange(`M${p.row}`).values=[['不适用']];else formula(rounds,'M'+p.row,`=I${p.row}/F${p.row}`);
  }
  rounds.getRange(`C2:E${plan.rounds.length+1}`).setNumberFormat('#,##0');
  rounds.getRange(`H2:I${plan.rounds.length+1}`).setNumberFormat('#,##0');
  rounds.getRange(`F2:G${plan.rounds.length+1}`).setNumberFormat('#,##0.000000');
  rounds.getRange(`J2:W${plan.rounds.length+1}`).setNumberFormat('#,##0.000000');
  rounds.getRange('A1:A1').format.columnWidth=27;rounds.getRange('B1:B1').format.columnWidth=17;
  rounds.getRange('C1:E1').format.columnWidth=11;rounds.getRange('F1:W1').format.columnWidth=19;
  rounds.getRange('X1:X1').format.columnWidth=70;rounds.getRange('Y1:AA1').format.columnWidth=68;rounds.getRange('AB1:AB1').format.columnWidth=20;
  rounds.getRange('AC1:AF1').format.columnWidth=22;
  rounds.getRange(`AC2:AF${plan.rounds.length+1}`).setNumberFormat('0.000000');
  title(summary,2,'Temporal：全部baseline与LG统一时间、能量及功率',14);
  put(summary,3,[['每轮2048次调用＝512个原TEST视频循环4遍。每16视频等效64帧。正式准确度沿用原完整TEST。']]);
  put(summary,4,[['baseline逐视频严格串行，无指令图或预取。36个合格点：12配置各三轮；40GB与80GB分平台，非受控同卡比较。']]);
  put(summary,6,[['平台','模型','MPS/%','合格轮次','每16视频/ms','等效单帧/ms','每16视频/J','等效单帧/J','共同tokens/s','自身tokens/s','调用合计','硬件功率上限/W','同区间实测板卡均功率/W','固定主机代理/W']]);
  header(summary,6,14);
  for(const g of plan.groups){
    const r=g.summaryRow,lo=g.lo,hi=g.hi;
    put(summary,r,[[platform(g.platform),labels[g.model],g.cap,g.rounds,null,null,null,null,null,null,null,g.hardware_power_limit_w,null,g.host_proxy_w]]);
    const sums=c=>`SUM('轮次原值'!${c}${lo}:${c}${hi})`;
    for(const [c,f] of [['E',`=${sums('F')}*1000/${sums('E')}*16`],['F',`=E${r}/64`],['G',`=${sums('G')}/${sums('E')}*16`],['H',`=G${r}/64`],['I',`=${sums('H')}/${sums('F')}`],['K',`=${sums('E')}`]])formula(summary,c+r,f);
    if(g.pooled_native_tokens_per_s===null)summary.getRange('J'+r).values=[['不适用']];else formula(summary,'J'+r,`=${sums('I')}/${sums('F')}`);
    formula(summary,'M'+r,`=${sums('AD')}/${sums('F')}`);
  }
  summary.getRange('A1:A1').format.columnWidth=29;summary.getRange('B1:B1').format.columnWidth=24;
  summary.getRange('C1:E1').format.columnWidth=21;summary.getRange('F1:N1').format.columnWidth=19;
  summary.getRange('L7:N18').setNumberFormat('0.000');
  summary.getRange('E7:J18').setNumberFormat('#,##0.000');summary.getRange('K7:K18').setNumberFormat('#,##0');
  const last=6+plan.groups.length;
  put(summary,last+2,[['共同速率＝合有效token数÷合保留秒数。自身token定义因模型不同不能互换。CLIP/YOLO自身文字token不适用。']]);
  put(summary,last+3,[['baseline能量为同区间NVML积分＋338.2W主机代理。LG采用固定组件代理，二者均非插座电表整机实测。']]);
  const lr=plan.lgInputStart;
  put(rounds,lr,[['LG独立审计处理值','数值','单位','含义']]);header(rounds,lr,4);
  const lg=data.LG;
  const values=[['实际异步软件',lg.measured_software_time16_ms,'ms/16视频','含原软件FFT、首场与全部电子建图'],
    ['实测电子＋固定光周期投影',lg.fixed_optics_counterfactual_time16_ms,'ms/16视频','完整DAG重排，非物理光路实测'],
    ['实际FFT窗板卡能量',lg.measured_software_board_FFT_energy16_j,'J/16视频','只作原软件诊断，不借作光路代理能量'],
    ['固定组件代理能量',lg.fixed_optics_component_proxy_energy16_j,'J/16视频','控制器、光器件、板卡各时间并集，重叠不重复'],
    ['投影共同有效token速率',lg.projected_common_tokens_per_s,'tokens/s','共同Qwen有效token分子'],
    ['投影自身有效token速率',lg.projected_native_tokens_per_s,'tokens/s','每16视频光场共享prompt实际词编号计一次，不复制16份'],
    ['前端与上场主干交叠',lg.actual_stream_span_overlap_ms,deployed?'ms/384场':'ms/128场','CUDA事件stream跨度，不证明逐kernel活跃并行'],
    ['实际软件共同token速率',lg.measured_software_common_tokens_per_s,'tokens/s','原共同分子除以实际软件连续窗'],
    ['实际软件自身token速率',lg.measured_software_native_tokens_per_s,'tokens/s','原共享prompt分子除以实际软件连续窗']];
  put(rounds,lr+1,values);rounds.getRange(`B${lr+1}:B${lr+9}`).setNumberFormat('#,##0.000000');
  title(summary,23,deployed?'LG三轮：不额外预热，电子图构建单列':'LG05异步：软件实测与光电时间投影',11);
  put(summary,25,[['口径','时间/ms每16视频','能量/J每16视频','共同tokens/s','自身tokens/s']]);header(summary,25,5);
  put(summary,26,[['实际异步软件',null,null,null,null],['电子＋固定光周期投影',null,null,null,null]]);
  formula(summary,'D26',`='轮次原值'!B${lr+8}`);formula(summary,'E26',`='轮次原值'!B${lr+9}`);
  formula(summary,'B26',`='轮次原值'!B${lr+1}`);formula(summary,'C26',`='轮次原值'!B${lr+3}`);
  formula(summary,'B27',`='轮次原值'!B${lr+2}`);formula(summary,'C27',`='轮次原值'!B${lr+4}`);
  formula(summary,'D27',`='轮次原值'!B${lr+5}`);formula(summary,'E27',`='轮次原值'!B${lr+6}`);
  summary.getRange('B26:E27').setNumberFormat('#,##0.000');
  put(summary,29,[['异步软件能量只指原FFT窗NVML板卡诊断。固定光周期行是改变传播时长后的DAG投影，不是光路计时或电表测量。']]);
  put(summary,30,[[phase?.scope_version==='exclude_software_CPU_submit_v2'?'新口径同时剔除纯FFT/ROI/电残差/H2D的CPU提交；保留电子建图、融合、merger、decoder及观测排队。':'原ROI和电残差在投影中置零，电子capture、融合、merger、任务头及必要等待保留。']]);
  put(summary,31,[['LG自身token按每16视频光场的实际共享prompt记一次，不乘16。不能与baseline自身token分子偷换。']]);
  summary.getRange('H25:K25').values=[['LG功率项','数值/W','性质','适用区间']];header(summary,25,5);
  summary.getRange('H25:K25').format={fill:'#526D82',font:{bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:56};
  const powers=lg.component_energy_proxy.powers_w;
  summary.getRange('H26:K30').values=[['硬件功率上限',lg.hardware_power_limit_w,'限额，非实测','A10040GB PCIe'],
    ['软件FFT窗板卡均功率',lg.measured_software_matched_board_average_w,'NVML实测','原软件连续计时窗'],
    ['固定控制器',powers.controller,'组件代理，非实测','控制器时间并集'],
    ['固定光器件',powers.optical_devices,'组件代理，非实测','光周期时间并集'],
    ['固定电子板卡',powers.electronic_board,'组件代理，非实测','电子时间并集']];
  summary.getRange('H26:K30').format={wrapText:true,rowHeight:42};summary.getRange('I26:I30').setNumberFormat('0.000');
  for(const r of [29,30,31]){summary.getRange(`A${r}:G${r}`).merge();summary.getRange(`A${r}:G${r}`).format={wrapText:true,rowHeight:42};}
  put(summary,32,[['LG06在初始排外Guard拒收，无有效计时或输出，未用于本表；事件池候选尚无GPU验证。']]);
  if(deployed){
    put(summary,30,[['v3仅在v2上排除每轮44个电子图构建host节点。首场、融合、merger、decoder、快照和观测队列保留。']]);
    put(summary,32,[['LG为100% MPS/108SM，三轮共6144调用；baseline沿用原边界。比较是软件实测对部署期光电投影，非实测光路百倍。']]);
  }
  summary.getRange('A32:K32').merge();summary.getRange('A32:K32').format={wrapText:true,rowHeight:32};
  put(summary,33,[['baseline相对LG','MPS/%','相对实际软件时间比','相对投影时间比','两种代理能量比','平台关系']]);header(summary,33,6);
  for(const [i,g] of plan.groups.entries()){
    const r=34+i;put(summary,r,[[labels[g.model],g.cap,null,null,null,g.platform===lg.platform?'同40GB平台':'跨配置，非受控同卡']]);
    formula(summary,'C'+r,`=E${g.summaryRow}/$B$26`);formula(summary,'D'+r,`=E${g.summaryRow}/$B$27`);formula(summary,'E'+r,`=G${g.summaryRow}/$C$27`);
  }
  summary.getRange(`C34:E${33+plan.groups.length}`).setNumberFormat('0.000"x"');
  summary.getRange(`F34:F${33+plan.groups.length}`).format={wrapText:true,rowHeight:36};
  if(deployed){
    put(summary,33,[['baseline相对LG','MPS/%','软件时间比','部署投影时间比','代理能效比','共同吞吐比','系统代理均功率比','平台关系']]);header(summary,33,8);
    for(const [i,g] of plan.groups.entries()){
      const r=34+i;
      formula(summary,'F'+r,`=$D$27/I${g.summaryRow}`);
      formula(summary,'G'+r,`=(G${g.summaryRow}/E${g.summaryRow})/($C$27/$B$27)`);
      summary.getRange('H'+r).values=[[g.platform===lg.platform?'同40GB平台':'跨配置，非受控同卡']];
    }
    summary.getRange('C34:G45').setNumberFormat('0.000"x"');
    summary.getRange('C34:G45').format.horizontalAlignment='right';summary.getRange('H34:H45').format={wrapText:true,rowHeight:42};
    const tr=lr+13;
    put(rounds,tr,[['LG轮次','视频调用','原软件总秒','原v2总秒','电子图构建/ms','v3投影总秒','组件代理/J','共同tokens','自身tokens','FFT板卡/J','FFT板卡均W','原运行根']]);header(rounds,tr,12);
    for(const [i,r] of data.LG_rounds.entries())put(rounds,tr+1+i,[[r.repeat,r.calls,r.software_elapsed_s,r.previous_v2_elapsed_s,r.graph_build_host_ms,
      r.projected_elapsed_s,r.component_proxy_energy_j,r.common_effective_tokens,r.native_effective_tokens,r.software_board_energy_j,r.software_matched_board_average_w,path.basename(path.dirname(r.path))]]);
    rounds.getRange(`C${tr+1}:K${tr+3}`).setNumberFormat('0.000000');rounds.getRange(`H${tr+1}:I${tr+3}`).setNumberFormat('#,##0');
    rounds.getRange(`L${tr+1}:L${tr+3}`).format={wrapText:true,rowHeight:55};
    const sum=c=>`SUM(${c}${tr+1}:${c}${tr+3})`,forms={1:`=${sum('C')}*1000/${sum('B')}*16`,2:`=${sum('F')}*1000/${sum('B')}*16`,
      3:`=${sum('J')}/${sum('B')}*16`,4:`=${sum('G')}/${sum('B')}*16`,5:`=${sum('H')}/${sum('F')}`,6:`=${sum('I')}/${sum('F')}`,
      8:`=${sum('H')}/${sum('C')}`,9:`=${sum('I')}/${sum('C')}`};
    for(const [offset,f] of Object.entries(forms))formula(rounds,'B'+(lr+Number(offset)),f);
    // Use explicit quoted ranges for cross-sheet pooled mean power.
    formula(summary,'I27',`=SUM('轮次原值'!J${tr+1}:J${tr+3})/SUM('轮次原值'!C${tr+1}:C${tr+3})`);
  }
  title(summary,46,'baseline关键节点分解（ms/16视频）',13);
  put(summary,48,[['模型','MPS/%','平台',...stages.map(([,label])=>label)]]);header(summary,48,13);
  for(const [i,g] of plan.groups.entries()){
    const r=49+i;put(summary,r,[[labels[g.model],g.cap,g.platform===lg.platform?'40GB/250W':'80GB/400W',...stages.map(()=>null)]]);
    for(const [j,[key]] of stages.entries()){
      if(g.stage_equivalent16_ms[key]===undefined)summary.getRange(col(j+3)+r).values=[['未独立记录']];
      else formula(summary,col(j+3)+r,`=SUM('轮次原值'!${col(j+13)}${g.lo}:${col(j+13)}${g.hi})/SUM('轮次原值'!E${g.lo}:E${g.hi})*16`);
    }
  }
  summary.getRange(`D49:M${48+plan.groups.length}`).setNumberFormat('0.000');
  put(summary,61,[['主时间＝文字分词＋入口调度＋完整GPU模型至评分＋中间CPU返回＋视频间调度。末条CPU返回在主窗外另列。']]);
  put(summary,62,[['图像准备、原始CPU到GPU搬运确实逐视频执行，仅在核算中排除。首网络节点分解是完整GPU模型行的子项，不重复相加。']]);
  put(summary,63,[['lab首网络节点为原首TF，YOLO为首网络层。租机没有独立首TF记录，不伪造该分解。']]);
  put(summary,64,[['完整GPU模型包括原图像前端、文字embedding、全部神经网络、完整词表和任务头，不只算网络主干。']]);
  put(summary,66,[[deployed?'LG GPU跨度（三轮共384场）':'LG GPU各资源跨度（完整128场）','原CUDA事件跨度合计/ms','性质']]);header(summary,66,3);
  const resourceLabels={optical_simulation:'软件光传播FFT',electronic_residual:'电子残差',electronics:'融合、merger及任务头',detector_ROI:'探测器ROI',frontend:'图文前端',H2D:'原始H2D搬运'};
  let sr=67;for(const [kind,value] of Object.entries(lg.GPU_resource_span_totals_NOT_additive))put(summary,sr++,[[resourceLabels[kind]??kind,value,'跨度有交叠，不能直接相加']]);
  if(deployed){
    title(summary,74,'LG等效单位与历史参考',6);
    put(summary,75,[['口径','每视频/ms','等效单帧/ms','每视频/J','等效单帧/J','定义']]);header(summary,75,6);
    for(const [r,source,label] of [[76,26,'实际软件（FFT板卡诊断能量）'],[77,27,'部署投影（组件代理能量）']]){
      put(summary,r,[[label,null,null,null,null,'一场16视频/64帧，摊销等效非独立帧延迟']]);
      for(const [c,f] of [['B',`=B${source}/16`],['C',`=B${source}/64`],['D',`=C${source}/16`],['E',`=C${source}/64`]])formula(summary,c+r,f);
    }
    put(summary,79,[['旧表LG05投影/ms每16视频',data.LG_historical.fixed_optics_counterfactual_time16_ms,'旧v2含建图，历史参考，不混入新三轮']]);
    summary.getRange('B76:E77').setNumberFormat('0.000000');summary.getRange('A76:F77').format={wrapText:true,rowHeight:44};
  }
  const indexRows=data.raw_file_index.map(r=>[r.role,r.local_path,r.remote_path??'不适用，本地生成',r.bytes,r.sha256,'原SHA/size重核']);
  put(index,1,[['文件类型','本地绝对路径','远端原绝对路径','字节数','SHA256','核验']]);put(index,2,indexRows);header(index,1,6);
  index.freezePanes.freezeRows(1);index.freezePanes.freezeColumns(1);
  index.getRange('A1:A1').format.columnWidth=29;index.getRange('B1:C1').format.columnWidth=100;
  index.getRange('D1:D1').format.columnWidth=17;index.getRange('E1:E1').format.columnWidth=68;index.getRange('F1:F1').format.columnWidth=23;
  index.getRange(`D2:D${indexRows.length+1}`).setNumberFormat('#,##0');
  // Fit actual multi-table column widths, long provenance paths and N/A markers.
  summary.getRange('C7:N18').format.horizontalAlignment='right';
  summary.getRange('B26:E27').format.horizontalAlignment='right';
  summary.getRange('B26:E26').format.wrapText=true;summary.getRange('A26:E26').format.rowHeight=40;
  summary.getRange(`D49:M${48+plan.groups.length}`).format.horizontalAlignment='right';
  summary.getRange('B67:B72').setNumberFormat('0.000');summary.getRange('B67:B72').format.horizontalAlignment='right';
  rounds.getRange(`C2:W${plan.rounds.length+1}`).format.horizontalAlignment='right';
  rounds.getRange(`X2:X${plan.rounds.length+1}`).format={wrapText:true,rowHeight:66};
  index.getRange(`B2:C${indexRows.length+1}`).format={wrapText:true,rowHeight:66};
  if(deployed)index.getRange(`A2:A${indexRows.length+1}`).format.wrapText=true;
  index.getRange(`D2:D${indexRows.length+1}`).format.horizontalAlignment='right';
  w.recalculate();
  if(physical){close(physical.getRange('C30').values[0][0],data.LG.fixed_optics_counterfactual_time16_ms);close(physical.getRange('C32').values[0][0],0);
    for(let r=45;r<45+plan.groups.length;r++)close(physical.getRange('J'+r).values[0][0],0);}
  for(const g of plan.groups){const r=g.summaryRow;for(const [cell,value] of [['E',g.time16_ms],['G',g.energy16_j],['I',g.pooled_common_tokens_per_s]])close(summary.getRange(cell+r).values[0][0],value);}
  const g=plan.groups[0],source=rounds.getRange('F'+g.lo),original=source.values[0][0],before=summary.getRange('E'+g.summaryRow).values[0][0];
  source.values=[[original+1]];w.recalculate();close(summary.getRange('E'+g.summaryRow).values[0][0]-before,1000*16/g.video_calls);
  source.values=[[original]];w.recalculate();close(summary.getRange('E'+g.summaryRow).values[0][0],before);
  if(deployed){
    const c=rounds.getRange('F'+(lr+14)),saved=c.values[0][0],before=summary.getRange('B27').values[0][0];
    c.values=[[saved+.1]];w.recalculate();close(summary.getRange('B27').values[0][0]-before,100/384);
    c.values=[[saved]];w.recalculate();close(summary.getRange('B27').values[0][0],before);
  }
  const error=/^#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)/;
  for(const [s,range] of [[summary,'A1:N80'],[rounds,`A1:AF${plan.lgInputStart+(deployed?17:10)}`]]){
    if(s.getRange(range).values.flat().some(x=>typeof x==='string'&&error.test(x)))throw Error('Formula error in '+s.name);
  }
  console.log((await w.inspect({kind:'table',range:`结果与阶段!A6:K${last}`,include:'values,formulas',tableMaxRows:11,tableMaxCols:11,maxChars:1600})).ndjson);
  console.log((await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:1400})).ndjson);
  await fs.mkdir(out,{recursive:true});const target=path.join(out,phase?'Temporal_LG与全部baseline统一时间能量功率_20261027.xlsx':'Temporal_三轮baseline与LG异步_时间能耗明细_20261026.xlsx');
  await (await SpreadsheetFile.exportXlsx(w)).save(target);
  for(const [sheetName,range,name] of [['结果与阶段',`A2:N${last+3}`,'summary'],['结果与阶段',`A23:K${33+plan.groups.length}`,'LG'],['结果与阶段',`A46:M${48+plan.groups.length}`,'stages'],['结果与阶段',deployed?'A61:K79':'A61:K72','LG_resources'],['轮次原值','A1:M8','rounds'],['轮次原值',`A${lr}:H${lr+9}`,'LG_inputs'],['原始文件索引','A1:F5','index'],...(deployed?[['轮次原值',`A${lr+13}:L${lr+16}`,'LG_three_rounds']]:[])]){
    if(refreshUnsealed&&!['LG_resources','index','LG_three_rounds'].includes(name))continue;
    const preview=await w.render({sheetName,range,scale:1.3,format:'png'});await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await preview.arrayBuffer()));
  }
  if(physical){
    for(const [range,name] of [['A2:E39','physical_LG'],['A42:L58','physical_baseline']]){
      if(refreshUnsealed&&name!=='physical_LG')continue;
      const preview=await w.render({sheetName:'物理阶段',range,scale:1.3,format:'png'});await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await preview.arrayBuffer()));
    }
    await fs.copyFile(phasePath,path.join(out,'phase_ledger.json'));
  }
  const csv=x=>'"'+String(x??'').replaceAll('"','""')+'"';
  await fs.writeFile(path.join(out,'raw_file_index.csv'),'\uFEFF'+[['role','local_path','remote_path','bytes','sha256'],...data.raw_file_index.map(r=>[r.role,r.local_path,r.remote_path,r.bytes,r.sha256])].map(r=>r.map(csv).join(',')).join('\r\n'));
  await fs.copyFile(input,path.join(out,'collected_results.json'));
  const checks={status:'passed',input_summary_sha256:await sha(input),workbook_sha256:await sha(target),lab_points:24,LG_rounds:deployed?3:1,all_rounds_retained:true,recalculation_input_mutation_restored:true,saved_file_independent_audit_performed_by_builder:false,visual_review_performed_by_builder:false};
  await fs.writeFile(path.join(out,'builder_checks.json'),JSON.stringify(checks,null,2));
  console.log(JSON.stringify({target,sha256:checks.workbook_sha256,visual_review_required:true,independent_saved_cache_audit_required:true}));
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const [input,out,phase,refresh]=process.argv.slice(2);if(!input||!out)throw Error('Usage: node build_energy_summary_20261026.mjs qualified.json NEW_OUTPUT_DIRECTORY [phase_ledger.json]');
  if(refresh&&refresh!=='--refresh-unsealed')throw Error('Unknown refresh option');
  await build(path.resolve(input),path.resolve(out),phase?path.resolve(phase):null,refresh==='--refresh-unsealed');
}
