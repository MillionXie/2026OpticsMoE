from pathlib import Path
import fitz,hashlib
out=Path('tmp/pdfs/timing_audit');out.mkdir(parents=True,exist_ok=True)
base=Path(r'C:\Users\Xml12\OneDrive\2026OpticsModel\实验设备\00-SLM-Meadowlark Optics\Blink Plus')
files=[(next(base.glob('*SLM7930.pdf')),[0]),(base/'PCIe User Manual.pdf',[18,19,29]),
       (Path('ABO_Lab_SHS_8um/vendor/manuals/SHS 硬件使用手册 20260309.pdf'),[18,19,24])]
for i,(path,pages) in enumerate(files):
 print('\nFILE',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
 doc=fitz.open(path);print('PAGES',len(doc))
 for p in pages:
  page=doc[p];print('\nPDF PAGE',p+1,'\n',page.get_text())
  if i==0 or (i==1 and p==29) or (i==2 and p==24):page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/f'{i}_p{p+1}.png')
