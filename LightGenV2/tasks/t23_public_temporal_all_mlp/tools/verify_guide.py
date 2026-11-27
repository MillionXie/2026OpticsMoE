"""Verify guide contents and preservation of the author's finalized formatting."""
from pathlib import Path
import argparse
import json
import zipfile

from docx import Document
from lxml import etree
from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[4]
PROJECT = REPO/'LightGen-II-MLP'
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def canonical(element):
    return etree.tostring(element, method='c14n') if element is not None else None


def verify(reference):
    document = Document(PROJECT/'README.docx')
    expected = [
        'data/input_for_review.npz', 'data/reference_mos.csv', 'weights/LightGen2.pt',
        'model.py', 'infer.py', 'visualize.py', 'requirements.txt', 'README.pdf',
    ]
    first = [row.cells[0].text for row in document.tables[0].rows]
    if first != expected:
        raise ValueError('The Files table differs from the actual project')
    for name in first:
        if not (PROJECT/name).is_file():
            raise FileNotFoundError(name)
    results = [row.cells[0].text for row in document.tables[1].rows]
    actual = sorted(path.name for path in (PROJECT/'results').iterdir())
    if results != actual or len(results) != 3:
        raise ValueError('The results table differs from the result folder')
    metrics = json.loads((PROJECT/'results/02_metrics.json').read_text('utf-8'))
    if not .795 <= metrics['SRCC'] <= .799:
        raise ValueError('The finalized rounded score is not supported')
    reader = PdfReader(PROJECT/'README.pdf')
    pages = [page.extract_text() for page in reader.pages]
    combined = ' '.join('\n'.join(pages).split()).replace(' -', '-').replace('- ', '-')
    for name in expected+results:
        if name not in combined:
            raise ValueError('A listed file is missing from the PDF '+name)
    for required in ('SRCC) of 0.80', 'Qwen3-VL', 'SRCC 0.77', 'two-layer global MLP'):
        if required not in combined:
            raise ValueError('A required result or description is missing '+required)
    for removed in ('03_prediction_vs_mos', 'software reference', 'Input optical field',
                    'NVIDIA', 'runtime_seconds', 'MLP version'):
        if removed in combined:
            raise ValueError('An obsolete guide item remains '+removed)
    for index, content in enumerate(pages, 1):
        if str(index) not in (line.strip() for line in content.splitlines()):
            raise ValueError('A visible page number differs '+str(index))
    with zipfile.ZipFile(reference) as original, zipfile.ZipFile(PROJECT/'README.docx') as current:
        editable = {'word/document.xml', 'word/_rels/document.xml.rels'}
        for name in original.namelist():
            if name in editable or name.startswith('word/media/'):
                continue
            if original.read(name) != current.read(name):
                raise ValueError('A preserved template part differs '+name)
        old = list(etree.fromstring(original.read('word/document.xml')).find('w:body', NS))
        new = list(etree.fromstring(current.read('word/document.xml')).find('w:body', NS))
        for index in (0,1,3,4,5,6,7,8,10,12,14):
            if canonical(old[index]) != canonical(new[index]):
                raise ValueError('An unchanged paragraph differs '+str(index))
        if canonical(old[-1]) != canonical(new[-1]):
            raise ValueError('The page geometry differs from the reference')
        if canonical(old[25]) != canonical(new[21]) or canonical(old[26]) != canonical(new[22]):
            raise ValueError('The author license paragraphs differ')
        for old_index, new_index in ((2,2),(13,13),(15,15),(16,16),(17,17),(24,20)):
            for part in ('w:pPr', 'w:r/w:rPr'):
                if canonical(old[old_index].find(part, NS)) != canonical(new[new_index].find(part, NS)):
                    raise ValueError('A changed paragraph was reformatted '+str(new_index))
        for table_index in (9,11):
            for part in ('w:tblPr','w:tblGrid'):
                if canonical(old[table_index].find(part, NS)) != canonical(new[table_index].find(part, NS)):
                    raise ValueError('A table layout differs '+str(table_index))
        media = [name for name in current.namelist() if name.startswith('word/media/')]
        if len(media) != 1 or current.read(media[0]) != (PROJECT/'results/03_input_frames.png').read_bytes():
            raise ValueError('The guide image differs from the actual input preview')
        fields = current.read('word/footer1.xml')
        if b'PAGE' not in fields:
            raise ValueError('The original page-number field is missing')
    report = dict(PDF_pages=len(reader.pages), source_file_rows=len(first),
                  result_file_rows=len(results), template_parts_preserved=True,
                  guide_images=1, SRCC=metrics['SRCC'], displayed_SRCC='0.80')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    verify(parser.parse_args().reference)
