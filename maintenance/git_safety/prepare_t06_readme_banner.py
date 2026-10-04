"""Add a current-version pointer without replacing T06 historical README text."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from prepare_scoped_main import git

PATH = "LightGenV2/tasks/t06_video_quality_assessment/README.md"
BANNER = ("\n**2026-10-04整理覆盖：**下文保留历史仿真、baseline和测速协议，不代表所有默认命令已适配。\n"
          "正式Spatial/Temporal实拍版本、各自权重及已发布核心代码见\n"
          "[当前版本与待收敛边界](CURRENT_VERSION_20261004.md)。主线核心通过57项CPU合同测试、\n"
          "两份正式PT严格加载；旧Temporal-36默认profile和后续实拍微调入口仍须单独核验。\n")


def prepare(root, base, expected_sha, message):
    root = root.resolve()
    main = git(root, "rev-parse", "main").decode().strip()
    head = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "merge-base", "--is-ancestor", main, base)
    if "branch refs/heads/main\n" in git(root, "worktree", "list", "--porcelain").decode():
        raise RuntimeError("main is checked out")
    index = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").decode().strip())
    before = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    original = git(root, "show", base + ":" + PATH)
    if hashlib.sha256(original).hexdigest() != expected_sha:
        raise RuntimeError("Historical README changed; re-review")
    title, body = original.decode("utf-8").split("\n", 1)
    if not title.startswith("# T06 ") or "2026-10-04整理覆盖" in body:
        raise RuntimeError("Unexpected title/already has banner")
    result = (title + "\n" + BANNER + "\n" + body).encode("utf-8")
    oid = git(root, "hash-object", "-w", "--stdin", data=result).decode().strip()
    with tempfile.TemporaryDirectory(prefix="lightgen_t06_doc_index_") as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))
        git(root, "read-tree", base, env=env)
        git(root, "update-index", "--cacheinfo", "100644", oid, PATH, env=env)
        changed = git(root, "diff", "--cached", "--name-only", base, env=env).decode().splitlines()
        if changed != [PATH]:
            raise RuntimeError("Unexpected documentation changes")
        tree = git(root, "write-tree", env=env).decode().strip()
        candidate = git(root, "commit-tree", tree, "-p", base, env=env, data=(message + "\n").encode()).decode().strip()
    after = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    if before != after or main != git(root, "rev-parse", "main").decode().strip() or head != git(root, "rev-parse", "HEAD").decode().strip():
        raise RuntimeError("Concurrent index/ref change")
    return {"candidate_commit": candidate, "base_main": main, "candidate_parent": base,
            "preserved_original_readme_sha256": expected_sha,
            "all_original_body_text_retained": True, "new_readme_sha256": hashlib.sha256(result).hexdigest(),
            "working_files_changed": False, "refs_updated": False, "pushed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[2], args.base_commit, args.expected_sha256,
                             "Point T06 historical entry to verified published runtime and remaining exceptions"), indent=2))
