import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const dir=path.dirname(fileURLToPath(import.meta.url));
const d=JSON.parse(await fs.readFile(path.join(dir,'statistics.json'),'utf8'));
const w=Workbook.create(),s=w.worksheets.add('Summary'),q=w.worksheets.add('Paired titles');
for(const sh of [s,q]) sh.showGridLines=false;
s.tabColor='#16835F';
s.getRange('A2').values=[['ABO text-to-image: adapted physical Ours vs frozen Qwen-2048D']];
s.getRange('A4:G4').values=[['Display category','Title queries','Ours correct','Qwen correct','Ours Hit@1','Qwen Hit@1','Gap (pp)']];
s.getRange('A5:A16').values=d.categories.map(v=>[v.category]);
q.getRange('A1:N1').values=[['Label','Product ID','Display category','Ours first rank','Qwen first rank','Ours Hit@1','Qwen Hit@1','Ours Hit@5','Qwen Hit@5','Ours Hit@10','Qwen Hit@10','Ours RR','Qwen RR','Official title']];
q.getRange('A2:N101').values=d.queries.map(v=>[v.title_label,v.product_id,v.category,v.ours_first_rank,v.qwen_first_rank,v.ours_hit1,v.qwen_hit1,v.ours_hit5,v.qwen_hit5,v.ours_hit10,v.qwen_hit10,v.ours_reciprocal_rank,v.qwen_reciprocal_rank,v.title]);
q.getRange('P1').values=[['Ours: raw/ours_cpu_replay.json (fixed readout/cache CPU replay, not original saved predictions).']];
q.getRange('P2').values=[['Qwen: raw/qwen_predictions.json and raw/qwen_report.json; titles/test CSV SHA identical.']];
q.getRange('P3').values=[['Display groups: raw/category_mapping.csv. Relevance remains exact SKU.']];
q.getRange('P4').values=[['Historical MRR/mAP: raw/ours_report.json; raw/qwen_report.json (full ranking).']];
s.getRange('B5:G16').formulas=d.categories.map((v,i)=>{const r=i+5;return[
  `=COUNTIFS('Paired titles'!$C$2:$C$101,A${r})`,
  `=SUMIFS('Paired titles'!$F$2:$F$101,'Paired titles'!$C$2:$C$101,A${r})`,
  `=SUMIFS('Paired titles'!$G$2:$G$101,'Paired titles'!$C$2:$C$101,A${r})`,
  `=C${r}/B${r}`,`=D${r}/B${r}`,`=(E${r}-F${r})*100`];});
s.getRange('A17').values=[['Overall']];
s.getRange('B17:G17').formulas=[['=SUM(B5:B16)','=SUM(C5:C16)','=SUM(D5:D16)','=C17/B17','=D17/B17','=(E17-F17)*100']];
s.getRange('A20:E20').values=[['Metric','Ours correct','Qwen correct','Ours rate','Qwen rate']];
s.getRange('A21:A23').values=d.overall.map(v=>[v.metric]);
s.getRange('B21:E23').formulas=[
  ["=SUM('Paired titles'!F2:F101)","=SUM('Paired titles'!G2:G101)",'=B21/$B$17','=C21/$B$17'],
  ["=SUM('Paired titles'!H2:H101)","=SUM('Paired titles'!I2:I101)",'=B22/$B$17','=C22/$B$17'],
  ["=SUM('Paired titles'!J2:J101)","=SUM('Paired titles'!K2:K101)",'=B23/$B$17','=C23/$B$17']];
s.getRange('A26:C26').values=[['Historical full metrics','Ours','Qwen']];
s.getRange('A27:C28').values=d.full_metrics.filter(v=>['mrr','map'].includes(v.metric)).map(v=>[v.metric,v.ours,v.qwen]);
s.getRange('A30').values=[['100 title queries; 2400 TEST images; 24 exact-SKU positives per title.']];
s.getRange('A31').values=[['Ours: 10-epoch readout adaptation, validation-selected epoch 9; Qwen is frozen 2048D.']];
s.getRange('A32').values=[['Twelve manual display groups. n means titles; small groups are descriptive, not repeated trials.']];
s.getRange('A33').values=[['Historical TEST had development exposure. This is cached-feature replay, not a fresh CCD acquisition.']];
for(const [sh,range,heads] of [[s,'A4:G33',['A4:G4','A20:E20','A26:C26']],[q,'A1:N101',['A1:N1']]]){
  sh.getRange(range).format.font={name:'Arial',size:11};sh.getRange(range).format.rowHeight=25;
  sh.getRange(range).format.verticalAlignment='center';sh.getRange(range).format.wrapText=false;
  for(const h of heads) sh.getRange(h).format={fill:'#264653',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:29,horizontalAlignment:'center',verticalAlignment:'center'};
}
s.getRange('A2').format.font={name:'Arial',size:16,bold:true};
s.getRange('A4:A33').format.columnWidth=26;s.getRange('B4:G33').format.columnWidth=18;
s.getRange('E5:F17').setNumberFormat('0.00%');s.getRange('G5:G17').setNumberFormat('+0.00;-0.00;0.00');
s.getRange('D21:E23').setNumberFormat('0.00%');s.getRange('B27:C28').setNumberFormat('0.0000');
s.getRange('A17:G17').format.fill='#EAF2ED';s.getRange('A17:G17').format.font.bold=true;
s.getRange('A30:A33').format.font={name:'Arial',size:11,color:'#52616B'};
q.getRange('A1:A101').format.columnWidth=10;q.getRange('B1:B101').format.columnWidth=18;
q.getRange('C1:C101').format.columnWidth=26;q.getRange('D1:M101').format.columnWidth=17;
q.getRange('L2:M101').setNumberFormat('0.0000');q.getRange('N1:N101').format.columnWidth=95;
q.getRange('N2:N101').format.wrapText=true;q.getRange('A2:N101').format.rowHeight=45;
q.getRange('P1:P4').format.columnWidth=115;q.freezePanes.freezeRows(1);
w.recalculate();
for(let i=0;i<12;i++){const v=d.categories[i],got=s.getRange(`B${i+5}:F${i+5}`).values[0],expected=[v.queries,v.ours_correct,v.qwen_correct,v.ours_accuracy,v.qwen_accuracy];if(got.some((x,j)=>Math.abs(x-expected[j])>1e-10))throw new Error('Category mismatch');}
if(JSON.stringify(s.getRange('B17:D17').values[0])!==JSON.stringify([100,85,82]))throw new Error('Totals mismatch');
const old=q.getRange('F2').values[0][0];q.getRange('F2').values=[[1-old]];w.recalculate();
if(s.getRange('C17').values[0][0]!==85+1-2*old)throw new Error('Recalculation failed');
q.getRange('F2').values=[[old]];w.recalculate();
const inspect=await w.inspect({kind:'table',range:'Summary!A4:G17',include:'values,formulas',tableMaxRows:14,tableMaxCols:7,maxChars:11000});await fs.writeFile(path.join(dir,'workbook_inspect.ndjson'),inspect.ndjson);
const errors=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'Formula error scan'});await fs.writeFile(path.join(dir,'formula_errors.ndjson'),errors.ndjson);
for(const [sheetName,range,name] of [['Summary','A1:G33','excel_summary_preview.png'],['Paired titles','A1:H7','excel_titles_preview.png']]){const img=await w.render({sheetName,range,scale:1.5,format:'png'});await fs.writeFile(path.join(dir,name),new Uint8Array(await img.arrayBuffer()));}
await(await SpreadsheetFile.exportXlsx(w)).save(path.join(dir,'ABO_T2I_statistics.xlsx'));
console.log('Verified 100 queries, Ours 85 vs Qwen 82 correct; 12 display groups.');
