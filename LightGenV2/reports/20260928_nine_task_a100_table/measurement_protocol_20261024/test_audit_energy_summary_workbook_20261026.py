"""Synthetic cache maps/XML only, not authoring or measuring a workbook."""
import unittest
from unittest.mock import patch
import audit_energy_summary_workbook_20261026 as auditor


def fixture():
    groups=[];points=[];summary={};rounds={};row=2
    lg=dict(measured_software_time16_ms=100.,measured_software_board_FFT_energy16_j=9.,
            fixed_optics_counterfactual_time16_ms=10.,fixed_optics_component_proxy_energy16_j=1.,
            projected_common_tokens_per_s=10000.,projected_native_tokens_per_s=100.,
            hardware_power_limit_w=250,measured_software_matched_board_average_w=90.,
            measured_software_common_tokens_per_s=1000.,measured_software_native_tokens_per_s=10.,
            component_energy_proxy=dict(powers_w=dict(controller=41.388,optical_devices=80.358,electronic_board=62.842)))
    for i,(model,n) in enumerate([('qwen',2),('clip',1)]):
        ps=[]
        for repeat in range(1,n+1):
            seconds=100.*repeat;energy=400.*seconds
            p=dict(platform='lab',model=model,cap=50,repeat=repeat,calls=2048,elapsed_s=seconds,
                energy_j=energy,common_effective_tokens=913408,native_effective_tokens=None if model=='clip' else 913408,
                hardware_power_limit_w=250,measured_board_energy_j=seconds*61.8,host_proxy_w=338.2,
                time16_ms=seconds*1000/128,energy16_j=energy/128,
                stage_totals_ms={key:seconds*1000/5 for key in auditor.STAGES[2:7]})
            ps.append(p);points.append(p)
            for c,value in [('E',2048),('F',seconds),('G',energy),('J',p['time16_ms']),
                            ('K',p['energy16_j']),('L',913408/seconds)]:rounds[c+str(row)]=str(value)
            if model!='clip':rounds['M'+str(row)]=str(913408/seconds)
            for c,value in [('AC',250),('AD',seconds*61.8),('AE',61.8),('AF',338.2)]:rounds[c+str(row)]=str(value)
            row+=1
        seconds=sum(p['elapsed_s'] for p in ps);energy=sum(p['energy_j'] for p in ps);calls=2048*n
        g=dict(platform='lab',model=model,cap=50);groups.append(g)
        values=dict(E=seconds*1000/calls*16,F=seconds*1000/calls/4,G=energy/calls*16,
                    H=energy/calls/4,I=913408*n/seconds,K=calls,L=250,M=61.8,N=338.2)
        if model!='clip':values['J']=values['I']
        for c,value in values.items():summary[c+str(i+7)]=str(value)
        for j,key in enumerate(auditor.STAGES):
            if key in ps[0]['stage_totals_ms']:
                summary[auditor.column(j+3)+str(i+49)]=str(sum(p['stage_totals_ms'][key] for p in ps)*16/calls)
        for c,value in [('C',values['E']/100),('D',values['E']/10),('E',values['G'])]:summary[c+str(i+34)]=str(value)
    for cell,key in [('B26','measured_software_time16_ms'),('C26','measured_software_board_FFT_energy16_j'),
                     ('B27','fixed_optics_counterfactual_time16_ms'),('C27','fixed_optics_component_proxy_energy16_j'),
                     ('D27','projected_common_tokens_per_s'),('E27','projected_native_tokens_per_s')]:summary[cell]=str(lg[key])
    for cell,value in [('I26',250),('I27',90.),('I28',41.388),('I29',80.358),('I30',62.842)]:summary[cell]=str(value)
    summary['D26']='1000';summary['E26']='10'
    return dict(groups=groups,rounds=points,LG=lg),[summary,rounds,{}]


class WorkbookAuditTests(unittest.TestCase):
    def test_expected_saved_cache_maps_pass(self):
        data,sheets=fixture();auditor.verify_values(data,sheets)

    def test_mean_of_rates_is_not_pooled_rate(self):
        data,sheets=fixture();sheets[0]['I7']=str((913408/100+913408/200)/2)
        with self.assertRaises(ValueError):auditor.verify_values(data,sheets)

    def test_swapped_actual_and_projected_energy_rejected(self):
        data,sheets=fixture();sheets[0]['C26'],sheets[0]['C27']=sheets[0]['C27'],sheets[0]['C26']
        with self.assertRaises(ValueError):auditor.verify_values(data,sheets)

    def test_wrong_phase_or_round_cache_rejected(self):
        for index,cell in [(0,'F49'),(1,'J2'),(0,'M7'),(1,'AE2'),(0,'I27'),(0,'I29')]:
            data,sheets=fixture();sheets[index][cell]='999999'
            with self.assertRaises(ValueError):auditor.verify_values(data,sheets)

    def test_missing_and_nonfinite_cache_rejected(self):
        for value in (None,'nan','inf'):
            data,sheets=fixture();sheets[0]['E7']=value
            with self.assertRaises(ValueError):auditor.verify_values(data,sheets)

    def test_formula_without_saved_value_rejected(self):
        ns=auditor.NS['s']
        workbook=f'<workbook xmlns="{ns}"><sheets>'+''.join(f'<sheet name="{name}"/>' for name in
                   ['结果与阶段','轮次原值','原始文件索引'])+'</sheets></workbook>'
        sheet=f'<worksheet xmlns="{ns}"><sheetData><row><c r="A1"><f>1+1</f></c></row></sheetData></worksheet>'
        class FakeZip:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,name):return workbook if name=='xl/workbook.xml' else sheet
        with patch.object(auditor,'ZipFile',return_value=FakeZip()):
            with self.assertRaises(ValueError):auditor.cached_sheets('not_a_file')


if __name__=='__main__':unittest.main()
