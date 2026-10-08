from pathlib import Path

import pytest

from prepare_payload_ignore_rules import reviewed_block, RULES


def test_extracts_only_reviewed_block_not_legacy_rules():
    source = (Path(__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")
    block = reviewed_block(source)
    actual = {line for line in block.splitlines() if line and not line.startswith("#")}
    assert actual == RULES
    assert "code_packages" not in block
    assert "openmoji_grid_v2" not in block
    assert "text_to_image/dataset" not in block


def test_rejects_unreviewed_source_hiding_rule():
    source = (Path(__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")
    source = source.replace("node_modules/", "node_modules/\n*.py")
    with pytest.raises(RuntimeError, match="Unexpected ignore rules"):
        reviewed_block(source)


def test_rejects_duplicate_publication_marker():
    source = (Path(__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly one"):
        reviewed_block(source + source)
