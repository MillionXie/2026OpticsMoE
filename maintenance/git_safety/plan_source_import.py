"""Read a pinned Git tree and plan missing Python dependencies; no file/Git writes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def plan(root: Path, ref: str, entries: list[str], include_contents: bool = False) -> dict:
    head = git(root, "rev-parse", "--verify", ref).decode().strip()
    tree = set(git(root, "ls-tree", "-r", "--name-only", head).decode().splitlines())
    sources, conflicts, missing, visited = [], [], [], set()
    queue = list(entries)

    def resolve(module: str) -> str | None:
        prefix = module.replace(".", "/")
        for candidate in (prefix + ".py", prefix + "/__init__.py"):
            if candidate in tree:
                return candidate
        return None

    while queue:
        relative = queue.pop(0)
        if relative in visited:
            continue
        visited.add(relative)
        if relative not in tree:
            missing.append(relative)
            continue
        data = git(root, "show", f"{head}:{relative}")
        content = data.decode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        target = root / relative
        if not target.resolve().is_relative_to(root.resolve()):
            conflicts.append({"path": relative, "reason": "outside repository"})
            continue
        if target.exists():
            local = target.read_text(encoding="utf-8")
            if local.replace("\r\n", "\n") != content.replace("\r\n", "\n"):
                conflicts.append({"path": relative, "reason": "different existing local file; never overwrite"})
        else:
            row = {"path": relative, "source_blob_sha256": digest}
            if include_contents:
                row["content"] = content
            sources.append(row)
        if not relative.endswith(".py"):
            continue
        package = relative[:-3].replace("/", ".").split(".")
        if not relative.endswith("/__init__.py"):
            package.pop()
        # Include package initializers; importing them can have dependencies too.
        for size in range(1, len(package) + 1):
            initializer = "/".join(package[:size]) + "/__init__.py"
            if initializer in tree and initializer not in visited:
                queue.append(initializer)
        for node in ast.walk(ast.parse(content, filename=relative)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package[:len(package) - node.level + 1]
                    if node.module:
                        parts += node.module.split(".")
                    module = ".".join(parts)
                else:
                    module = node.module or ""
                modules = [module, *(module + "." + alias.name for alias in node.names)]
            for module in modules:
                dependency = resolve(module)
                if dependency and dependency not in visited:
                    queue.append(dependency)
    return {"source_head": head, "entries": entries, "new_files": sources,
            "existing_conflicts": conflicts, "missing_entries": missing,
            "visited_paths": sorted(visited), "static_imports_only": True,
            "runtime_verified": False, "read_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--include-contents", action="store_true")
    args = parser.parse_args()
    report = plan(Path(__file__).resolve().parents[2], args.ref, args.entry, args.include_contents)
    print(json.dumps(report, ensure_ascii=True))
    raise SystemExit(bool(report["existing_conflicts"] or report["missing_entries"]))


if __name__ == "__main__":
    main()
