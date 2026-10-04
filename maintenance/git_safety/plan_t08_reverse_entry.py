"""Pin the real reverse T08 entry under an unambiguous name, without copying it."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'd0662a7d240340817948a3496c2cd43f4240e76d'
TASK = 'LightGenV2/tasks/t08_abo_image_text_retrieval/'


def main():
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', SOURCE+':'+TASK+'optical_moe.py'])
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'ebf026e5aa2eaf6dc32ef7b5c5f18c5ac2eaea8515fe2d5a3431427d67e97869':
        raise RuntimeError('Actual reverse runtime identity changed')
    target = ROOT/'maintenance/storage/T08_REVERSE_ENTRY_PLAN_20261004.json'
    if target.exists():
        raise FileExistsError(target)
    plan = dict(source_commit=SOURCE, task_prefix=TASK,
                paths=[dict(source_path=TASK+'optical_moe.py', target_path=TASK+'reverse_runtime.py',
                            sha256=digest, bytes=len(raw))],
                purpose='Preserve actual text-to-image implementation separately from existing image-to-title entry.',
                source_rewritten=False, runtime_verified=False, published=False,
                pending=['Explicit reverse-direction launcher and portable configuration',
                         '10cm shared backend strict PT and synthetic compatibility',
                         'Original Windows deployment dependency closure'])
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(source_sha256=digest, bytes=len(raw), published=False)))


if __name__ == '__main__':
    main()
