"""Read-only verification of saved XLSX data and preservation of user content."""
import json
import math
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from .verify_final_plot_workbooks import NS, cells, col
from .prepare_final_plot_data import sha

TASK = Path(__file__).resolve().parent
FOLDER = TASK / 'runs/outputs/t19_final_plot_data_20261024'


def load(path):
    with ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(t.text or '' for t in row.findall('.//m:t', NS)) for row in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        pages = [cells(z, f'xl/worksheets/sheet{i}.xml', shared) for i in (1, 2)]
        samples = ET.fromstring(z.read('xl/worksheets/sheet2.xml'))
        # Import/export normalizes transient UI selections. Compare functional
        # view attributes, panes and exact column definitions instead.
        view = samples.find('m:sheetViews/m:sheetView', NS)
        structure = {'cols': ET.tostring(samples.find('m:cols', NS)),
                     'view': {k:v for k,v in view.attrib.items() if k != 'tabSelected'},
                     'panes': [dict(p.attrib) for p in view.findall('m:pane', NS)]}
        names = [s.get('name') for s in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', NS)]
    return pages, structure, names


def equal(actual, expected):
    if isinstance(expected, (int, float)):
        assert actual is not None and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-13), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    payload = json.loads((TASK/'runs/outputs/t19_plot_supplement_20261025/data.json').read_text(encoding='utf-8'))
    reports = []
    for key, name in [('abo', '04_ABO图搜图.xlsx'), ('salicon', '07_SALICON显著性.xlsx')]:
        file = FOLDER/name
        prior, structure, names = load(FOLDER/'verification'/f'{key}_before_plot_supplement.bin')
        current, after_structure, after_names = load(file)
        assert names == after_names == ['整体指标', '逐测试样本']
        assert {a:v for a,v in prior[1].items() if v is not None} == {a:v for a,v in current[1].items() if v is not None}
        assert structure == after_structure, 'Original sample-sheet columns or functional panes changed'
        for address, value in prior[0].items():
            if value is not None:
                equal(current[0].get(address), value)
        if key == 'abo':
            rows = payload['abo_categories']['rows']
            for i, row in enumerate(rows, 17):
                for j, value in enumerate(row, 1):
                    equal(current[0].get(col(j)+str(i)), value)
            assert sum(current[0]['C'+str(r)] for r in range(17, 27)) == 623
            assert sum(current[0]['B'+str(r)] for r in range(17, 27)) == 800
            equal(current[0]['D28'], 623/800)
            count = len(rows)
        else:
            rows = payload['saliency']['rows']
            for i, row in enumerate(rows, 6):
                for j, value in enumerate(row[:5], 7):
                    equal(current[0].get(col(j)+str(i)), value)
                # Main-table CC and ID must agree exactly with preserved old sheet.
                equal(current[0]['G'+str(i)], prior[1]['A'+str(i-4)])
                equal(current[0]['J'+str(i)], prior[1]['B'+str(i-4)])
            assert len({current[0]['G'+str(i)] for i in range(6, 5006)}) == 5000
            for letter, metric in [('H','mae'),('I','kld'),('J','cc'),('K','nss')]:
                mean = sum(current[0][letter+str(i)] for i in range(6, 5006))/5000
                assert abs(mean-payload['saliency']['reported_metrics'][metric]) < 2e-5
            count = len(rows)
        reports.append(dict(file=name, sha256=sha(file), supplemented_rows=count, saved_cells_verified=True,
                            original_main_values_preserved=True, original_sample_values_columns_and_panes_preserved=True,
                            transient_UI_selection_normalized_by_import_export=True))
    report = dict(status='passed', files=reports, new_neural_forwards=0, new_optimizer_steps=0)
    (FOLDER/'verification/plot_supplement_saved_xlsx_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
