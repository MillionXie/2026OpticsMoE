import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const dir=path.dirname(fileURLToPath(import.meta.url));
const d=JSON.parse(await fs.readFile(path.join(dir,'statistics.json'),'utf8'));
const w=Workbook.create();
const s=w.worksheets.add('Category comparison');
const q=w.worksheets.add('Paired queries');
for(const sh of [s,q]) sh.showGridLines=false;
s.tabColor='#16835F';
s.getRange('A2').values=[['ABO image-to-image: physical Ours vs frozen Qwen-64D']];
s.getRange('A4:H4').values=[['Category','Products','Queries','Ours correct','Qwen correct','Ours Top-1','Qwen Top-1','Gap (pp)']];
s.getRange('A5:B14').values=d.categories.map(v=>[v.category,v.products]);
q.getRange('A1:I1').values=[['Sample ID','Product ID','Category','Ours Hit@1','Qwen Hit@1','Ours first rank','Ours Top-1 SKU','Qwen Top-1 SKU','Dataset category']];
q.getRange('A2:I801').values=d.queries.map(v=>[v.sample_id,v.product_id,v.category,v.ours_hit1,v.qwen_hit1,v.ours_first_relevant_rank,v.ours_top1_product_id,v.qwen_top1_product_id,v.dataset_category]);
q.getRange('K1').values=[['Source: raw/ours_report.json; raw/qwen_predictions.csv; raw/dataset_manifest.csv; raw/protocol.json.']];
q.getRange('K2').values=[['Qwen Top-5/10 totals: raw/qwen_report.json (not exposed by the paired Top-1 CSV).']];
s.getRange('C5:H14').formulas=d.categories.map((v,i)=>{const a=i+5;return [
  `=COUNTIFS('Paired queries'!$C$2:$C$801,A${a})`,
  `=SUMIFS('Paired queries'!$D$2:$D$801,'Paired queries'!$C$2:$C$801,A${a})`,
  `=SUMIFS('Paired queries'!$E$2:$E$801,'Paired queries'!$C$2:$C$801,A${a})`,
  `=D${a}/C${a}`,`=E${a}/C${a}`,`=(F${a}-G${a})*100`];});
s.getRange('A15').values=[['Overall']];
s.getRange('B15:H15').formulas=[['=SUM(B5:B14)','=SUM(C5:C14)','=SUM(D5:D14)','=SUM(E5:E14)','=D15/C15','=E15/C15','=(F15-G15)*100']];
s.getRange('A18:F18').values=[['Metric','Queries','Ours correct','Qwen correct','Ours rate','Qwen rate']];
s.getRange('A19:D21').values=d.overall.map(v=>[v.metric,v.queries,v.ours_correct,v.qwen_correct]);
s.getRange('E19:F21').formulas=d.overall.map((_,i)=>[`=C${i+19}/B${i+19}`,`=D${i+19}/B${i+19}`]);
s.getRange('A23').values=[['Exact-SKU matching; each category contains 20 products and 80 held-out queries.']];
s.getRange('A24').values=[['Ours: physical gallery and query. Qwen: frozen model, original RGB inputs, 64D.']];
s.getRange('A25').values=[['Historical TEST-selected model; same-spin enrolled-product views, not independent new-product generalization.']];
for(const [sh,used,heads] of [[s,'A4:H25',['A4:H4','A18:F18']],[q,'A1:I801',['A1:I1']]]) {
  sh.getRange(used).format.font={name:'Arial',size:11};
  sh.getRange(used).format.rowHeight=25;
  sh.getRange(used).format.verticalAlignment='center';
  sh.getRange(used).format.wrapText=false;
  for(const header of heads) sh.getRange(header).format={fill:'#264653',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:29,horizontalAlignment:'center',verticalAlignment:'center'};
}
s.getRange('A2').format.font={name:'Arial',size:16,bold:true};
s.getRange('A4:A25').format.columnWidth=21;
s.getRange('B4:E21').format.columnWidth=17;
s.getRange('F4:H21').format.columnWidth=18;
s.getRange('F5:G15').setNumberFormat('0.00%');
s.getRange('H5:H15').setNumberFormat('+0.00;-0.00;0.00');
s.getRange('E19:F21').setNumberFormat('0.000%');
s.getRange('A15:H15').format.fill='#EAF2ED';
s.getRange('A15:H15').format.font.bold=true;
s.getRange('A23:A25').format.font={name:'Arial',size:11,color:'#52616B'};
q.getRange('A1:A801').format.columnWidth=23;
q.getRange('B1:C801').format.columnWidth=19;
q.getRange('D1:F801').format.columnWidth=18;
q.getRange('G1:I801').format.columnWidth=21;
q.getRange('K1:K2').format.columnWidth=110;
q.freezePanes.freezeRows(1);
w.recalculate();
for(let i=0;i<10;i++) {
  const got=s.getRange(`C${i+5}:G${i+5}`).values[0];
  const v=d.categories[i],expected=[v.queries,v.ours_correct,v.qwen_correct,v.ours_accuracy,v.qwen_accuracy];
  if(got.some((x,j)=>Math.abs(x-expected[j])>1e-10)) throw new Error('Category mismatch '+v.category);
}
if(JSON.stringify(s.getRange('B15:E15').values[0])!==JSON.stringify([200,800,649,681])) throw new Error('Total mismatch');
// Test a simple source edit, then restore before export.
const previous=q.getRange('D2').values[0][0];
q.getRange('D2').values=[[1-previous]];
w.recalculate();
if(s.getRange('D15').values[0][0]!==649+1-2*previous) throw new Error('Recalculation failed');
q.getRange('D2').values=[[previous]];
w.recalculate();
const check=await w.inspect({kind:'table',range:'Category comparison!A4:H15',include:'values,formulas',tableMaxRows:12,tableMaxCols:8,maxChars:11000});
await fs.writeFile(path.join(dir,'workbook_inspect.ndjson'),check.ndjson);
const errors=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'Formula error scan'});
await fs.writeFile(path.join(dir,'formula_errors.ndjson'),errors.ndjson);
for(const [sheetName,range,name] of [['Category comparison','A1:H25','excel_summary_preview.png'],['Paired queries','A1:I8','excel_query_preview.png']]) {
  const img=await w.render({sheetName,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(dir,name),new Uint8Array(await img.arrayBuffer()));
}
await(await SpreadsheetFile.exportXlsx(w)).save(path.join(dir,'ABO_I2I_statistics.xlsx'));
console.log('Verified 200 products, 800 paired queries; correct counts 649 and 681.');
