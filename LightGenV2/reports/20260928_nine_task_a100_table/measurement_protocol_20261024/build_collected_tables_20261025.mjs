/** Author new, explicitly partial workbooks from sealed audited indices. No models. */
import fs from 'node:fs/promises';
import fss from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const HERE=path.dirname(fileURLToPath(import.meta.url));
const REPO=path.resolve(HERE,'../../../..');
const OUT=path.join(HERE,'outputs/01a061e2-6f88-7913-87f5-b632000e4c83/collected_20261025_01');
const norm=p=>p.replaceAll('\\','/');
const rel=p=>norm(path.relative(REPO,p));
const read=p=>JSON.parse(fss.readFileSync(p,'utf8'));
const shaCache=new Map();
function sha(p){p=path.resolve(p);if(!shaCache.has(p))shaCache.set(p,crypto.createHash('sha256').update(fss.readFileSync(p)).digest('hex'));return shaCache.get(p);}
function assert(x,s){if(!x)throw Error(s);}
function close(a,b,s){assert(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=Math.max(1e-8,Math.abs(b)*1e-9),`${s}: ${a} != ${b}`);}
const MAIN=path.join(HERE,'measured_index_partial_17.json'),TEMP=path.join(HERE,'temporal_matrix_partial_07.json');
assert(sha(MAIN)==='5d05ea0c395b2dfdc07135b4e8cf15957c3171568af466cda7a83fb2c4593bda','Sealed main index differs');
assert(sha(TEMP)==='3b439f3cb1b6f9b61a0349147304247259d3514e767265b6ea5a368ff34c2b1b','Sealed temporal index differs');
const main=read(MAIN),temp=read(TEMP),p=read(path.join(HERE,'protocol.json'));
const t07='LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/';
const t06='LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/';
const names={'01a':'LGVQ 时间质量','01b':'LGVQ 空间质量','02':'ABO 图搜文','03':'ABO 图搜图','04':'ABO 文搜图','05':'LSP 关键点','06':'SALICON 显著性','07':'OpenMoji 语义交互','08':'T12 图像生成'};
const fullN={'01a':558,'01b':558,'02':2400,'03':800,'04':100,'05':1000,'06':5000,'07':1000,'08':2304};
const mainRecords=[],mainRuns=[],tempRecords=[],tempRuns=[],checks=[],fileRows=[],originalPins=new Map();
const check=(name,value,detail='')=>{assert(value,name+' '+detail);checks.push({name,status:'通过',detail});};

function mainFolders(row,sys){
  if(!row[sys])return [];
  if(row.id==='03')return [1,2,3].map(i=>t07+(sys==='ours'?'a100_input_inclusive_20261024_01/':'a100_input_inclusive_20261024_qwen01/')+`${sys}_repeat0${i}`);
  if(['05','06'].includes(row.id))return [1,2,3].map(i=>`LightGenV2/tasks/${row.id==='05'?'t02_keypoint_detection':'t03_saliency'}/runs/simulation/a100_input_inclusive_20261024_${row.id==='05'?'02':'01'}/${sys}_repeat0${i}`);
  if(row.id==='01b')return [1,2,3].map(i=>t06+`lgvq_spatial_input_a100_20261024_01/qwen_repeat0${i}`);
  if(row.id==='04'&&sys==='qwen')return [1,2,3].map(i=>`LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/a100_title_input_inclusive_20261024_02/qwen_repeat0${i}`);
  return row[sys].raw_references.map(x=>norm(x.raw_run_path||x.path||''));
}
function costFromReport(folder){
  const root=path.join(REPO,folder),reportName=fss.existsSync(path.join(root,'summary.json'))?'summary.json':'report.json';
  const report=read(path.join(root,reportName)),c=report.cost||{};
  assert(report.status==='complete',`Complete original report required ${folder}`);
  const ms=c.sample_ms??c.ms_per_video??report.sample_ms??report.continuous_wall_s*5,
    j=c.sample_j??c.estimated_joules_per_video??report.sample_j??(report.total_system_energy_j??report.system_energy_j)/200;
  assert(ms>0&&j>0,`Positive cost required ${folder}: ${ms}, ${j}`);
  return {root,report,reportName,ms,j,seconds200:ms/5,joules200:j*200,
    optical_s:(report.optical_ms_total??0)/1000,
    charged_s:report.charged_electronic_ms_total==null?null:report.charged_electronic_ms_total/1000,
    board_j:c.gpu_energy_j??(report.optical_ms_total==null?j*200-338.2*ms/5:null)};
}
function opOf(x){
  if(x?.derived_NN_dense_MAC_subtotal)return x.derived_NN_dense_MAC_subtotal;
  if(x?.TOP_MAC_convention_per_query)return {OP600:x.TOP_MAC_convention_per_query*600e12,TOP_per_input:x.TOP_MAC_convention_per_query,TOPS:x.TOPS_MAC_convention,TOPS_per_W:x.TOPS_per_W_MAC_convention,not_complete_scalar_OP:true};
  return null;
}
function aggregate(x,sys,id){
  if(!x)return null;
  const a=x.aggregate||x;
  return {ms:a.sample_ms??a.total_ms,j:a.sample_j??a.phase_energy_j,
    tokens:x.common_input_tokens_total??a.common_input_tokens_total??null,
    op:opOf(x),scope:x.scope||''};
}
function registerPin(file,digest,bytes,origin){
  const absolute=path.resolve(file);assert(absolute.startsWith(REPO+path.sep),'File outside repository');
  if(originalPins.has(absolute))assert(originalPins.get(absolute).sha256===digest,'Conflicting original SHA '+absolute);
  originalPins.set(absolute,{sha256:digest,bytes,origin});
}
function loadPins(folder){
  // Original file manifest and data-only backup receipts are independent byte anchors.
  for(let dir=folder,depth=0;depth<3;depth++,dir=path.dirname(dir)){
    for(const name of fss.readdirSync(dir)){
      if(name!=='files_manifest.json'&&!/^backup_receipt.*\.json$/.test(name))continue;
      const f=path.join(dir,name);let value;try{value=read(f);}catch{continue;}
      const rows=Array.isArray(value)?value:value.verified_files;
      if(!Array.isArray(rows))continue;
      for(const z of rows){const n=z.file||z.path;if(!n||!z.sha256)continue;const target=path.resolve(dir,norm(n));
        if(target===folder||target.startsWith(folder+path.sep))registerPin(target,z.sha256,z.bytes,rel(f));
      }
    }
  }
}
function fileCatalog(folders){
  const unique=[...new Set(folders.map(norm))];
  for(const folder of unique){
    const absolute=path.join(REPO,folder);loadPins(absolute);
    for(const name of fss.readdirSync(absolute)){
      const f=path.join(absolute,name);if(!fss.statSync(f).isFile())continue;
      const pin=originalPins.get(path.resolve(f));const digest=sha(f),bytes=fss.statSync(f).size;
      if(pin){assert(digest===pin.sha256,'Original SHA mismatch '+rel(f));if(pin.bytes!=null)assert(bytes===pin.bytes,'Original size mismatch '+rel(f));}
      fileRows.push({run:folder,file:rel(f),bytes,sha256:digest,
        verification:pin?'封存SHA及字节数已核对':'本地SHA登记',pin_source:pin?.origin||'',
        role:/backup_receipt|local_independent|audit/i.test(name)?'收据或独立复算':'原始运行文件'});
    }
  }
  return unique;
}
for(const row of main.nine_metric_rows){
  for(const sys of ['ours','qwen']){
    const x=row[sys],a=aggregate(x,sys,row.id),folders=mainFolders(row,sys),key=row.id+'|'+sys;
    const tokens=row.common_tokens?.total??a?.tokens??null;
    const record={key,id:row.id,task:names[row.id],system:sys,metric:row.metric,
      accuracy:row.full_test_reference[sys],full_test_n:fullN[row.id],
      ours_lab:sys==='ours'?row.full_test_reference.ours_lab??null:null,
      calls:a?600:null,tokens,op:a?.op||null,status:a?'已验收三轮':'待测',scope:a?.scope||'尚无合格成本'};
    if(a){
      assert(folders.length===3,'Exactly3 qualified repetitions');
      const runs=folders.map((folder,i)=>{const c=costFromReport(folder);return {key,task:names[row.id],system:sys,repeat:i+1,n:200,folder,...c};});
      close(runs.reduce((s,r)=>s+r.ms,0)/3,a.ms,key+' time');close(runs.reduce((s,r)=>s+r.j,0)/3,a.j,key+' energy');
      for(const r of runs){
        const refs=x.raw_references||x.repeats||row.raw_audit?.repeats.filter(z=>z.system===sys)||[];
        const expected=refs[r.repeat-1];const pin=expected?.report_sha256||expected?.summary_sha256||expected?.raw_summary_sha256;
        if(pin)registerPin(path.join(r.root,r.reportName),pin,null,'sealed main17 report digest');
        if(sys==='ours')close(r.joules200,121.746*r.optical_s+104.230*r.charged_s,key+' phase energy');
        mainRuns.push(r);
      }
      record.seconds=runs.reduce((s,r)=>s+r.seconds200,0);record.joules=runs.reduce((s,r)=>s+r.joules200,0);
      record.optical_s=runs.reduce((s,r)=>s+r.optical_s,0);record.charged_s=sys==='ours'?runs.reduce((s,r)=>s+r.charged_s,0):null;
      record.ms=a.ms;record.j=a.j;record.common_tps=tokens/record.seconds;
      if(row.common_tokens)close(record.common_tps,row.common_tokens[sys+'_per_s'],key+' common tokens/s');
      assert(record.op,'Qualified14 NN count required');
      close(record.op.OP600/600e12,record.op.TOP_per_input,key+' OP per input');
      close(record.op.OP600/1e12/record.seconds,record.op.TOPS,key+' TOPS');
      close(record.op.OP600/1e12/record.joules,record.op.TOPS_per_W,key+' TOPSW');
    }
    mainRecords.push(record);
  }
}
check('主表已验收成本单元',mainRecords.filter(r=>r.calls===600).length===14,'14/18');
check('主表完整配对行',main.nine_metric_rows.filter(r=>r.ours&&r.qwen).length===6,'6/9');
for(const row of [...temp.rows,{...temp.supplementary_rental_qwen_anchor,model:'qwen',cap:100,supplementary:true}]){
  assert(row.status==='complete'&&row.qualified_completed.length===3,'Qualified temporal triple required');
  const key=(row.platform.power_limit_w===400?'80GB':'40GB')+'|'+row.model+'|'+row.cap;
  const runs=row.qualified_completed.map((ref,i)=>{const folder=norm(ref.path),c=costFromReport(folder);registerPin(path.join(c.root,'report.json'),ref.report_sha256,null,'sealed temporal07 report digest');
    for(const [file,pin] of [['calls.csv',ref.calls_sha256],['telemetry.csv',ref.power_sha256]])if(pin)registerPin(path.join(c.root,file),pin,null,'sealed temporal07 byte digest');
    close(c.seconds200,ref.cost.continuous_wall_s,key+' raw wall');close(c.joules200,ref.cost.estimated_system_energy_j,key+' raw energy');
    close(c.joules200,338.2*c.seconds200+ref.cost.gpu_energy_j,key+' baseline energy');
    return {key,task:'LGVQ temporal',system:row.model,repeat:i+1,n:200,folder,...c};});
  const seconds=runs.reduce((s,r)=>s+r.seconds200,0),joules=runs.reduce((s,r)=>s+r.joules200,0),a=row.aggregate;
  close(seconds/600*1000,a.sample_ms,key+' aggregate time');close(joules/600,a.sample_j,key+' aggregate energy');
  close(a.sample_ms/4,a.equivalent_frame_ms,key+' equivalent frame');close(a.sample_j/4,a.equivalent_frame_j,key+' equivalent frame energy');
  const record={key,id:key,task:'LGVQ temporal',system:row.model,metric:'SRCC',accuracy:a.historical_full558_reference.srcc,full_test_n:558,
    calls:600,tokens:267600,op:opOf(a),status:'已验收三轮',platform:row.platform.gpu_name,power_limit:row.platform.power_limit_w,
    uuid:row.platform.gpu_uuid,cap:row.cap,seconds,joules,ms:a.sample_ms,j:a.sample_j,supplementary:!!row.supplementary};
  if(record.op){close(record.op.OP600/1e12/seconds,record.op.TOPS,key+' TOPS');close(record.op.OP600/1e12/joules,record.op.TOPS_per_W,key+' TOPSW');}
  tempRecords.push(record);tempRuns.push(...runs);
}
tempRecords.push({key:'40GB|ours|100',id:'ours',task:'LGVQ temporal',system:'ours',metric:'SRCC',accuracy:.8044,full_test_n:558,calls:null,tokens:267600,status:'待测',platform:'A100 PCIe 40GB',power_limit:250,cap:100,op:null});
check('多baseline正式成本轮数',tempRuns.length===39,'36 baseline +3 extra80GB Qwen');
const sharedMain=mainRuns.filter(r=>r.key==='01a|qwen').map(r=>norm(r.folder));
const sharedTemp=tempRuns.filter(r=>r.key==='40GB|qwen|100').map(r=>norm(r.folder));
check('两表共享40GB Qwen100原三轮',JSON.stringify(sharedMain)===JSON.stringify(sharedTemp));
const folders=fileCatalog([...mainRuns,...tempRuns].map(r=>r.folder));
check('去重合格点数量',folders.length===78,'42+39−3 shared=78');
check('已核原始SHA',fileRows.filter(r=>r.verification==='封存SHA及字节数已核对').length>500);
check('未对封存SHA的文件仅为清单本体或辅助收据',fileRows.filter(r=>r.verification!=='封存SHA及字节数已核对').every(r=>/^(backup_receipt.*\.json|local_independent.*\.json|files_manifest\.json)$/.test(path.basename(r.file))),'原始数据文件均有封存字节锚；清单不能包含自身SHA');
for(const row of main.nine_metric_rows.filter(r=>r.ours&&r.qwen)){
  const o=mainRecords.find(r=>r.key===row.id+'|ours'),q=mainRecords.find(r=>r.key===row.id+'|qwen');
  close(o.common_tps/q.common_tps,q.ms/o.ms,row.id+' commonTokens ratio');
}
check('全部配对Tokens率比等于时间反比',true,'六行逐项复算');
const payload={status:'PARTIAL_COLLECTED_DATA_NOT_FINAL_ALL18',created_beijing:'2026-10-25',
  main_index:{path:rel(MAIN),bytes:fss.statSync(MAIN).size,sha256:sha(MAIN)},
  temporal_index:{path:rel(TEMP),bytes:fss.statSync(TEMP).size,sha256:sha(TEMP)},
  mainRecords,tempRecords,mainRuns:mainRuns.map(({report,root,...r})=>r),tempRuns:tempRuns.map(({report,root,...r})=>r),
  checks,raw_files:fileRows,unique_qualified_points:folders.length,source_commit:execFileSync('git',['rev-parse','HEAD'],{cwd:REPO,encoding:'utf8'}).trim(),
  fixed_parameters:{optical_cycle_ms:1.0447,optical_phase_w:121.746,charged_electronic_w:104.230,baseline_host_proxy_w:338.2,max_power_age_s:.25},
  no_models_or_GPU_calls:true,no_qualified_measurement_replayed:true};
await fs.mkdir(OUT,{recursive:true});
await fs.writeFile(path.join(OUT,'collected_data_and_audit.json'),JSON.stringify(payload,null,2));
const csv=x=>'"'+String(x??'').replaceAll('"','""')+'"';
await fs.writeFile(path.join(OUT,'raw_file_index.csv'),'\ufeff'+['run,file,bytes,sha256,verification,pin_source,role',...fileRows.map(r=>Object.values(r).map(csv).join(','))].join('\n'));

const colors={navy:'#233B53',blue:'#E8F1FA',green:'#E7F2E8',fixed:'#FFF1D6',pending:'#FFF0D9',ink:'#202C38',muted:'#5E6A76'};
const col=i=>{let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
const scaledOp=v=>v==null?'未核定':v/600e12;
function sheet(w,name,title,headers,rows){
  const s=w.worksheets.add(name);s.showGridLines=false;
  s.getRange(`A1:${col(headers.length-1)}${rows.length+8}`).format.font={name:'Arial',size:11,color:colors.ink};
  s.getRange('A2').values=[[title]];s.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:colors.ink};
  s.getRange(`A5:${col(headers.length-1)}5`).values=[headers];s.getRange(`A5:${col(headers.length-1)}5`).format={fill:colors.navy,font:{color:'#FFFFFF',bold:true},wrapText:true,verticalAlignment:'center',horizontalAlignment:'center',rowHeight:43};
  if(rows.length)s.getRange(`A6:${col(headers.length-1)}${rows.length+5}`).values=rows;
  s.getRange(`A6:${col(headers.length-1)}${rows.length+5}`).format.rowHeight=28;
  s.getRange(`A5:${col(headers.length-1)}${rows.length+5}`).format.columnWidth=14;
  s.freezePanes.freezeRows(5);s.freezePanes.freezeColumns(2);s.tabColor=colors.navy;
  return s;
}
function formulas(s,row,items){for(const [column,formula] of Object.entries(items))s.getRange(column+row).formulas=[[formula]];}
function fmt(s,range,format){s.getRange(range).setNumberFormat(format);s.getRange(range).format.horizontalAlignment='right';}
function baseData(w,records,runs){
  const data=sheet(w,'指标与计算','完整TEST指标与成本计算',
    ['键','任务','系统','正式指标','完整TEST值','TEST样本数','已验收调用数','共同输入\nTokens总数','NN OP总数小计','状态','总秒数','总焦耳','ms/样本','J/样本','任务/s','Tokens/s','NN TOP/样本','NN TOPS小计','NN TOPS/W小计','光学dense等效OP','OP来源','serial NN TOP/样本','parallel NN TOP/样本','其他digital TOP/样本'],
    records.map(r=>[r.key,r.task,r.system,r.metric,r.accuracy,r.full_test_n,r.calls??'待测',r.tokens??'待核定',r.op?.OP600??(r.calls?'未核定':'待测'),r.status,null,null,null,null,null,null,null,null,null,r.system==='ours'?'未核定':'不适用',r.op?.local_receipt||r.op?.source||(r.op?rel(MAIN)+' [原封存NN小计字段]':'尚无合格算量收据'),scaledOp(r.op?.charged_serial_OP600??r.op?.charged_serial_NN_OP600??(r.system!=='ours'?r.op?.OP600:null)),r.system==='ours'?scaledOp(r.op?.parallel_residual_OP600??r.op?.parallel_residual_NN_OP600):'不适用',r.system==='ours'?scaledOp(r.op?.other_executed_digital_dense_OP600_not_in_NN_subtotal):'未单列']));
  data.getRange('B5:B30').format.columnWidth=24;data.getRange('A5:A30').format.columnWidth=24;data.getRange('D5:D30').format.columnWidth=27;data.getRange('G5:H30').format.columnWidth=21;data.getRange('I5:I30').format.columnWidth=29;data.getRange('J5:J30').format.columnWidth=18;data.getRange('U5:U30').format.columnWidth=72;
  data.getRange(`A6:X${records.length+5}`).format.rowHeight=56;
  data.getRange('U6:U30').format.wrapText=true;
  fmt(data,`E6:E${records.length+5}`,'0.0000');fmt(data,`F6:I${records.length+5}`,'#,##0');fmt(data,`K6:S${records.length+5}`,'0.000000');
  data.getRange(`E6:I${records.length+5}`).format.fill=colors.blue;data.getRange(`K6:S${records.length+5}`).format.fill=colors.green;
  const raw=sheet(w,'三轮原始数据','每轮200次 原始汇总与路径',
    ['键','任务','系统','轮次','样本数','总时间 s','总能耗 J','光学阶段 s','电子计费 s','板卡积分 J','原始报告','报告SHA256','计费口径'],
    runs.map(r=>[r.key,r.task,r.system,r.repeat,r.n,r.seconds200,r.joules200,r.optical_s,r.charged_s,r.board_j,r.folder+'/'+r.reportName,sha(path.join(REPO,r.folder,r.reportName)),r.system==='ours'?'实测分量加固定光学时间的混合投影':'实测连续墙钟；能量为主机代理加板卡积分']));
  raw.getRange('A5:A90').format.columnWidth=25;raw.getRange('B5:B90').format.columnWidth=23;raw.getRange('K5:K90').format.columnWidth=100;raw.getRange('L5:L90').format.columnWidth=70;
  fmt(raw,`F6:J${runs.length+5}`,'0.000000');raw.getRange(`F6:J${runs.length+5}`).format.fill=colors.blue;
  raw.getRange(`G6:G${runs.length+5}`).format.fill=colors.green;raw.getRange(`H6:H${runs.length+5}`).format.fill=colors.fixed;
  runs.forEach((r,i)=>{if(r.system==='ours')raw.getRange('F'+(i+6)).format.fill=colors.green;});
  raw.getRange(`M5:M${runs.length+5}`).format.columnWidth=55;
  fmt(data,`V6:X${records.length+5}`,'0.000000');data.getRange(`V6:X${records.length+5}`).format.fill=colors.green;
  for(let i=0;i<records.length;i++){
    const row=i+6,last=runs.length+5,guard=`G${row}=600`,total=field=>`SUMIFS('三轮原始数据'!$${field}$6:$${field}$${last},'三轮原始数据'!$A$6:$A$${last},A${row})`;
    formulas(data,row,{K:`=IF(${guard},${total('F')},"待测")`,L:`=IF(${guard},${total('G')},"待测")`,M:`=IF(${guard},K${row}/G${row}*1000,"待测")`,N:`=IF(${guard},L${row}/G${row},"待测")`,O:`=IF(${guard},G${row}/K${row},"待测")`,P:`=IF(${guard},H${row}/K${row},"待测")`,Q:`=IF(ISNUMBER(I${row}),I${row}/G${row}/10^12,"未核定")`,R:`=IF(ISNUMBER(I${row}),I${row}/K${row}/10^12,"未核定")`,S:`=IF(ISNUMBER(I${row}),I${row}/L${row}/10^12,"未核定")`});
  }
  data.getRange(`A6:U${records.length+5}`).conditionalFormats.addCustom('$J6="待测"',{fill:colors.pending});
  return data;
}
function archiveSheet(w,records,runs){
  const folders=new Set(runs.map(r=>r.folder)),files=fileRows.filter(r=>folders.has(r.run));
  const s=sheet(w,'原始文件索引','合格点原始文件 SHA256索引',
    ['所属点','文件相对仓库路径','字节数','SHA256','核查方式','封存清单来源','类型'],
    files.map(r=>Object.values(r)));
  s.getRange(`A5:B${files.length+5}`).format.columnWidth=95;s.getRange(`D5:D${files.length+5}`).format.columnWidth=70;s.getRange(`E5:G${files.length+5}`).format.columnWidth=28;
  s.getRange(`F5:F${files.length+5}`).format.columnWidth=95;
  s.getRange(`A6:G${files.length+5}`).format.rowHeight=80;
  for(const c of ['A','B','F'])s.getRange(`${c}6:${c}${files.length+5}`).format={wrapText:true,verticalAlignment:'center',font:{size:10}};
  const methods=sheet(w,'口径与审查','参数 证据边界与独立审查',
    ['项目','数值或结论','类型','说明'],[
      ['成本完成度',w===undefined?'':'主表14/18 多baseline36/36 额外80GB Qwen3/3','状态','尚有四成本单元待测'],
      ['颜色图例','蓝色 实测分量 绿色 推导 橙色 固定参数','说明','完整TEST值为原正式报告引用；原始数据计费口径另列'],
      ['单光学pass',1.0447,'固定参数 ms','.714+.300+.0307 不宣称新装置墙钟实测'],
      ['光学阶段功率',121.746,'固定参数 W','80.358+41.388'],['电子计费功率',104.230,'固定参数 W','62.842+41.388'],['baseline主机代理',338.2,'固定参数 W','另加同窗NVML板卡积分 非整机电表'],
      ['功率最大支持年龄',.25,'固定阈值 s','守卫没有放宽'],['三轮工作负载','first200 TEST ×3 fresh loaded process','科学合同','首条及199间隙保留 零dummy预热 非200次冷启动'],
      ['Ours时间与能量','实测前端和选定NN 加参数光学阶段','混合投影','非整机全软件墙钟 非已实测物理装置E2E'],
      ['baseline时间与能量','文件到CPU结果连续墙钟 同窗板卡积分','实测加主机代理','模型加载及独立离线图库不计'],
      ['光学等效算量','未核定','未知','不填0 不用最大容量或GPU FFT填充'],['NN算量','1 real MAC =2 OP','推导小计','不是完整scalarOP或硬件指令数'],
      ['共同Tokens','同批Qwen有效非padding输入数','共同工作量','不是生成吞吐 CLIP/YOLO同样用共同参考分子'],
      ['加速倍数','T参照 / T被比较系统','推导','大于1才表示被比较系统更快'],['节能倍数','E参照 / E被比较系统','推导','节能率另列为1−E被比较/E参照'],
      ['EDP改善倍数','T参照×E参照/(T被比较×E被比较)','推导','等于加速倍数×节能倍数'],
      ['主索引',rel(MAIN),'封存数据',sha(MAIN)],['Temporal索引',rel(TEMP),'封存数据',sha(TEMP)],
      ...checks.map(r=>[r.name,r.status,'独立检查',r.detail])]);
  methods.getRange('A5:A60').format.columnWidth=29;methods.getRange('B5:B60').format.columnWidth=66;methods.getRange('C5:C60').format.columnWidth=18;methods.getRange('D5:D60').format.columnWidth=95;
  methods.getRange('B22:B23').format={wrapText:true,rowHeight:56};
  methods.getRange('B8:B12').format.fill=colors.fixed;methods.getRange('B8:B12').setNumberFormat('0.000000');
  return {files,archive:s,methods};
}
async function finish(w,name,previewRanges){
  w.recalculate();
  const errors=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:3000});
  await fs.writeFile(path.join(OUT,name+'.formula_scan.ndjson'),errors.ndjson);
  for(const [sheetName,range] of previewRanges){const blob=await w.render({sheetName,range,scale:1.4,format:'png'});await fs.writeFile(path.join(OUT,name+'_'+sheetName+'.png'),new Uint8Array(await blob.arrayBuffer()));}
  const out=await SpreadsheetFile.exportXlsx(w);await out.save(path.join(OUT,name+'.xlsx'));
  const inspection=await w.inspect({kind:'table',range:previewRanges[0][0]+'!A5:P16',include:'values,formulas',tableMaxRows:12,tableMaxCols:16,maxChars:7000});
  await fs.writeFile(path.join(OUT,name+'.inspect.ndjson'),inspection.ndjson);
}
const w1=Workbook.create();
const summary=sheet(w1,'九行对比','A100 九行任务 已采集数据',
  ['编号','任务与原生单位','指标','Ours 完整TEST','Qwen 完整TEST','Ours 投影ms','Qwen 实测ms','加速倍数','Ours 阶段估算J','Qwen 代理加积分J','节能倍数','节能率','EDP改善倍数','Tokens/s倍率','成本状态'],
  main.nine_metric_rows.map(r=>[r.id,names[r.id]+(['01a','01b'].includes(r.id)?' 每视频':r.id==='04'?' 每标题':' 每样本'),r.metric,r.full_test_reference.ours,r.full_test_reference.qwen,null,null,null,null,null,null,null,null,null,r.ours&&r.qwen?'双方三轮齐全':r.id==='08'?'双方待测':'Ours待测']));
summary.getRange('A3').values=[['14/18成本单元 已有6行完整配对 其余3行保留待测 不以历史成本填充']];
summary.getRange('A6:A14').setNumberFormat('00');summary.getRange('B5:B20').format.columnWidth=31;summary.getRange('C5:C20').format.columnWidth=20;summary.getRange('D5:E20').format.columnWidth=19;summary.getRange('I5:J20').format.columnWidth=20;summary.getRange('O5:O20').format.columnWidth=24;
baseData(w1,mainRecords,mainRuns);
for(let i=0;i<9;i++){
  const n=i+6,o=6+i*2,q=o+1,cell=(r,c)=>`'指标与计算'!${c}${r}`;
  formulas(summary,n,{F:`=${cell(o,'M')}`,G:`=${cell(q,'M')}`,H:`=IF(AND(ISNUMBER(F${n}),ISNUMBER(G${n})),G${n}/F${n},"待测")`,I:`=${cell(o,'N')}`,J:`=${cell(q,'N')}`,K:`=IF(AND(ISNUMBER(I${n}),ISNUMBER(J${n})),J${n}/I${n},"待测")`,L:`=IF(ISNUMBER(K${n}),1-I${n}/J${n},"待测")`,M:`=IF(ISNUMBER(H${n}),H${n}*K${n},"待测")`,N:`=IF(ISNUMBER(H${n}),${cell(o,'P')}/${cell(q,'P')},"待测")`});
}
fmt(summary,'D6:E14','0.0000');fmt(summary,'F6:K14','0.000');fmt(summary,'L6:L14','0.00%');fmt(summary,'M6:N14','0.000');summary.getRange('F6:N14').format.fill=colors.green;
summary.getRange('A6:O14').conditionalFormats.addCustom('$O6<>"双方三轮齐全"',{fill:colors.pending});
summary.getRange('A17').values=[['OpenMoji采用epoch45 仿真0.8765 实验室0.8655 后者单列于方法说明 不混入成本比。']];
summary.getRange('A18').values=[['Ours为含输入的混合硬件投影 baseline为连续墙钟 能耗分别为阶段参数模型与主机代理加板卡积分。']];
archiveSheet(w1,mainRecords,mainRuns);
await finish(w1,'A100_九行任务_已采集核验_20261025',[['九行对比','A1:O18'],['指标与计算','A4:J23'],['指标与计算','K4:X23'],['三轮原始数据','A4:J13'],['原始文件索引','A4:G10'],['口径与审查','A4:D22']]);

const w2=Workbook.create();
const s2=sheet(w2,'Temporal对比','LGVQ Temporal 多baseline 已采集数据',
  ['模型','平台','MPS资源档','完整558 SRCC','ms/视频','J/视频','ms/等效帧','J/等效帧','相对同平台Qwen100加速','相对同平台Qwen100节能','节能率','EDP改善倍数','共同Tokens/s','Ours加速倍数','Ours节能倍数','状态'],
  tempRecords.map(r=>[r.system,r.power_limit===400?'SXM 80GB 400W':'PCIe 40GB 250W',r.cap/100,r.accuracy,null,null,null,null,null,null,null,null,null,'待Ours测量','待Ours测量',r.supplementary?'补充同机Qwen锚':r.status]));
s2.getRange('A3').values=[['36轮正式baseline +3轮80GB同机Qwen对照 Ours成本待测 不跨平台混称模型收益']];
s2.getRange('B5:B25').format.columnWidth=23;s2.getRange('I5:P25').format.columnWidth=22;
baseData(w2,tempRecords,tempRuns);
for(let i=0;i<tempRecords.length;i++){
  const n=i+6,d=i+6,r=tempRecords[i],anchor=tempRecords.findIndex(x=>x.system==='qwen'&&x.cap===100&&x.power_limit===r.power_limit)+6;
  formulas(s2,n,{E:`='指标与计算'!M${d}`,F:`='指标与计算'!N${d}`,G:`=IF(ISNUMBER(E${n}),E${n}/4,"待测")`,H:`=IF(ISNUMBER(F${n}),F${n}/4,"待测")`,I:`=IF(ISNUMBER(E${n}),E${anchor}/E${n},"待测")`,J:`=IF(ISNUMBER(F${n}),F${anchor}/F${n},"待测")`,K:`=IF(ISNUMBER(J${n}),1-F${n}/F${anchor},"待测")`,L:`=IF(ISNUMBER(I${n}),I${n}*J${n},"待测")`,M:`='指标与计算'!P${d}`});
}
fmt(s2,'C6:C19','0%');fmt(s2,'D6:D19','0.0000');fmt(s2,'E6:J19','0.000');fmt(s2,'K6:K19','0.00%');fmt(s2,'L6:M19','0.000');s2.getRange('E6:O19').format.fill=colors.green;s2.getRange('N6:O19').format.fill=colors.pending;
const cap=sheet(w2,'资源档对比','50档相对100档的成本变化',
  ['模型','平台','100档 ms/视频','50档 ms/视频','耗时倍率 50/100','100档 J/视频','50档 J/视频','能耗倍率 50/100','50档加速比 100/50'],
  ['qwen','glm','clip','yolo','deepseek','kimi'].map(m=>[m,['deepseek','kimi'].includes(m)?'80GB 400W':'40GB 250W',null,null,null,null,null,null,null]));
for(let i=0;i<6;i++){
  const n=i+6,a=6+i*2,b=a+1;formulas(cap,n,{C:`='Temporal对比'!E${a}`,D:`='Temporal对比'!E${b}`,E:`=D${n}/C${n}`,F:`='Temporal对比'!F${a}`,G:`='Temporal对比'!F${b}`,H:`=G${n}/F${n}`,I:`=C${n}/D${n}`});
}
fmt(cap,'C6:I11','0.000');cap.getRange('C6:I11').format.fill=colors.green;cap.getRange('B5:I15').format.columnWidth=23;
archiveSheet(w2,tempRecords,tempRuns);
await finish(w2,'A100_Temporal多基线_已采集核验_20261025',[['Temporal对比','A1:P19'],['指标与计算','A4:J19'],['指标与计算','K4:X19'],['三轮原始数据','A4:J13'],['资源档对比','A1:I13'],['原始文件索引','A4:G10'],['口径与审查','A4:D22']]);
console.log(JSON.stringify({status:payload.status,qualified_main:14,complete_pairs:6,temporal_trials:39,unique_points:folders.length,raw_files:fileRows.length,pinned_files:fileRows.filter(x=>x.verification==='封存SHA及字节数已核对').length,output:OUT}));
