import numpy as np
import four_image_flow as flow
from shs_physical2400 import SHSBench,CORNERS

OUT=flow.PROJECT/'runs/abo_i2i_20260926/shs_wait_sweep'


def main():
    flow.BASE_CORNERS=CORNERS.copy();OUT.mkdir(parents=True,exist_ok=False)
    source=flow.PROJECT/'runs/abo_i2i_20260926/shs_verify'
    flat=flow.PROJECT/'runs/abo_i2i_20260926/shs_phase_verify/flat.bmp'
    rows=[];refs={}
    with SHSBench(OUT,300,700,{}) as bench:
        for wait in (700,400,500,700):
            bench.wait=wait/1000
            for i in range(10):
                for name in ('marker','inverse'):
                    key=f'w{wait}_i{i}_{name}_{len(rows)}'
                    a,_=bench.capture('timing',flat,[source/(name+'.bmp')],[key],'flip_v')
                    if name not in refs:refs[name]=a[0]
                    row=dict(bench.rows[-1]);row['pcc_correct']=flow.pcc(a[0],refs[name])
                    if len(refs)==2:row['pcc_wrong']=flow.pcc(a[0],refs['inverse' if name=='marker' else 'marker'])
                    rows.append(row)
                    flow.write(OUT/'report.json',{'complete':False,'rows':rows})
    flow.write(OUT/'report.json',{'complete':True,'rows':rows})


if __name__=='__main__':main()
