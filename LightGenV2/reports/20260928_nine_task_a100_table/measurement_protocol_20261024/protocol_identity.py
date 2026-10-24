"""Apply an explicit human-approved identity amendment, never rewrite raw runs.

Keep the original protocol bytes and ID for existing immutable measurements.
Only the newly approved OpenMoji identity is overlaid; measurement boundaries
and every other task remain exactly equal to the original contract.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def effective_protocol(base_path=HERE/'protocol.json',
                       amendment_path=HERE/'protocol_amendment_openmoji_e45.json'):
    base_path, amendment_path = Path(base_path), Path(amendment_path)
    raw = base_path.read_bytes()
    base = json.loads(raw)
    amendment = json.loads(amendment_path.read_bytes())
    if (hashlib.sha256(raw).hexdigest() != amendment['base_protocol_sha256']
            or base['protocol_id'] != amendment['base_protocol_id']
            or amendment['status'] != 'human_approved_identity_correction'
            or amendment['row_id'] != '07'):
        raise ValueError('Identity amendment is not bound to the approved original protocol')
    rows = [row for row in base['rows'] if row['id'] == '07']
    if (len(rows) != 1
            or rows[0]['ours_pt_sha256'] != amendment['superseded_identity']['ours_pt_sha256']
            or rows[0]['ours_target'] != amendment['superseded_identity']['ours_target']):
        raise ValueError('Refuse an unrecognized or already-amended base identity')
    row = amendment['effective_row']
    if (row['id'] != '07' or row['ours_epoch'] != 45 or row['qwen_epoch'] != 30
            or row['layout_version'] != 'layered_anchor6_svg_v3'
            or row['ours_target'] != .8765 or row['ours_lab_target'] != .8655
            or row['qwen_target'] != .812 or row['full_test_samples'] != 1000
            or row['ours_pt_sha256'] != '03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21'
            or row['qwen_pt_sha256'] != '0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8'):
        raise ValueError('Not the human-selected laboratory .8655 epoch45 version')
    result = copy.deepcopy(base)
    result['rows'] = [copy.deepcopy(row) if item['id'] == '07' else item
                      for item in result['rows']]
    result['identity_amendments'] = [dict(
        id=amendment['amendment_id'], path=amendment_path.name,
        sha256=hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
        base_protocol_sha256=amendment['base_protocol_sha256'],
        identity_only=True, no_new_timing_result=True)]
    return result
