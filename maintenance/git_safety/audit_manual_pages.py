"""Read selected hardware-manual pages; this is not a measured timing result."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect_pages(pdf, pages, *, opener=None):
    pdf = Path(pdf)
    if not pages or any(p < 1 for p in pages) or len(set(pages)) != len(pages):
        raise ValueError('Use unique one-based page numbers')
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if opener is None:
        import fitz
        opener = fitz.open
    with opener(pdf) as doc:
        if any(p > len(doc) for p in pages):
            raise ValueError('Selected page outside manual')
        rows = [{'page': p, 'text': doc[p-1].get_text()} for p in pages]
        total = len(doc)
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Manual changed during inspection')
    return dict(manual=str(pdf), sha256=digest, total_pages=total, pages=rows,
                read_only=True, measured_timing=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--pages', type=int, nargs='+', required=True, help='One-based pages')
    args = parser.parse_args()
    print(json.dumps(inspect_pages(args.pdf, args.pages), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
