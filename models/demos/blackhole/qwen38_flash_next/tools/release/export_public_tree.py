"""Export the public tree of this model directory: the manifest's file list, scanned for internal tokens.

    python -m models.demos.blackhole.qwen38_flash_next.tools.release.export_public_tree --out /tmp/qwen38-public
    python -m ...export_public_tree --git-tree            # a tree object in this repo, no branch, sha on stdout
    python -m ...export_public_tree --check               # scan only

``tools/release/manifest.json`` names every public file, the pending files (needed by the public code, still carrying
the campaign's runtime pins; exported only with ``--with-pending``) and the forbidden token patterns.  Nothing is
rewritten: a forbidden token in a public file fails the export and every hit is listed.  A path with a ``dev``
directory component is never public or pending, and public code (``.py``, ``.sh``) may not import or name
``tools/dev`` or ``tests/dev``.  ``--with-dev`` adds ``tools/dev`` and ``tests/dev`` unscanned.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = MODEL_DIR.parents[3]
MANIFEST = Path(__file__).with_name("manifest.json")
DEV_DIRS = ("tools/dev/", "tests/dev/")
# the dotted form (``...tools.dev.x``), the path form with or without the model-directory prefix, and a ``"dev"`` path
# component built with pathlib
DEV_IMPORT = re.compile(r"\btools\.dev\b|(?<![A-Za-z0-9_])(?:tools|tests)/dev/|/\s*\"dev\"\s*/|\"dev\"\s*/")
DEV_IMPORT_SUFFIXES = {".py", ".sh"}
RELEASE_TOOLING = {
    "tools/release/export_public_tree.py",
    "tools/release/manifest.json",
    "tests/test_release_export_static.py",
}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".textproto", ".cpp", ".toml", ".yaml", ".yml", ".cfg"}


def is_dev_path(rel: str) -> bool:
    """A ``dev`` directory component anywhere in a model-relative path: never public, never pending."""

    return "dev" in Path(rel).parts[:-1]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    token_class: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.token_class}] {self.text.strip()[:160]}"


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key in ("public", "pending", "forbidden"):
        if key not in manifest:
            raise ValueError(f"manifest lacks {key!r}")
    return manifest


def tree_files(root: Path = MODEL_DIR) -> list[str]:
    """Every file of the model directory that is not dev-only, relative POSIX paths."""

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(DEV_DIRS):
            continue
        files.append(rel)
    return files


def scan(rel: str, patterns: dict[str, re.Pattern[str]], *, root: Path = MODEL_DIR) -> list[Finding]:
    path = root / rel
    if path.suffix not in TEXT_SUFFIXES or rel in RELEASE_TOOLING:  # the release tooling and its test name the tokens
        return []
    findings = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        for token_class, pattern in patterns.items():
            if pattern.search(line):
                findings.append(Finding(rel, number, token_class, line))
        if path.suffix in DEV_IMPORT_SUFFIXES and DEV_IMPORT.search(line):
            findings.append(Finding(rel, number, "dev-import", line))
    return findings


def audit(manifest: dict, *, root: Path = MODEL_DIR) -> tuple[list[Finding], list[Finding], list[str], list[str]]:
    """(public findings, pending findings, files in the tree the manifest does not classify, listed files missing).

    A listed path with a ``dev`` directory component is a public finding (``dev-path``) whether or not it exists."""

    patterns = {name: re.compile(pattern) for name, pattern in manifest["forbidden"].items()}
    public, pending = list(manifest["public"]), dict(manifest["pending"])
    present = set(tree_files(root))
    listed = set(public) | set(pending)
    unclassified = sorted(present - listed)
    missing = sorted(rel for rel in listed - present if not is_dev_path(rel))
    public_findings = [Finding(rel, 0, "dev-path", rel) for rel in sorted(listed) if is_dev_path(rel)]
    public_findings += [finding for rel in public if rel in present for finding in scan(rel, patterns, root=root)]
    pending_findings = [finding for rel in pending if rel in present for finding in scan(rel, patterns, root=root)]
    return public_findings, pending_findings, unclassified, missing


def export(files: list[str], out: Path, *, root: Path = MODEL_DIR, allow_dev: bool = False) -> None:
    denied = [rel for rel in files if is_dev_path(rel) and not (allow_dev and rel.startswith(DEV_DIRS))]
    if denied:
        raise ValueError(f"refusing to export dev paths: {denied}")
    for rel in files:
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)


def write_git_tree(files: list[str], *, root: Path = MODEL_DIR) -> str:
    """A tree object of the whole repo with the model directory replaced by the export; prints its sha, makes no ref."""

    prefix = root.relative_to(REPO_ROOT).as_posix()
    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        git = lambda *args: subprocess.run(
            ("git", "-C", str(REPO_ROOT), *args), check=True, env=env, stdout=subprocess.PIPE, text=True
        ).stdout  # noqa: E731
        git("read-tree", "HEAD")
        git("rm", "-r", "-q", "--cached", prefix)
        git("update-index", "--add", "--", *(f"{prefix}/{rel}" for rel in files))
        return git("write-tree").strip()


def dev_files(root: Path = MODEL_DIR) -> list[str]:
    return [
        path.relative_to(root).as_posix()
        for prefix in DEV_DIRS
        for path in sorted((root / prefix).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="directory to copy the public tree into (must not exist)")
    target.add_argument("--git-tree", action="store_true", help="write a git tree object instead of files")
    target.add_argument("--check", action="store_true", help="scan only")
    parser.add_argument(
        "--with-pending",
        action="store_true",
        help="also export the pending files (their findings are listed, not fatal)",
    )
    parser.add_argument("--with-dev", action="store_true", help="also export tools/dev and tests/dev (not scanned)")
    parser.add_argument("--strict", action="store_true", help="pending findings are fatal too")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    public_findings, pending_findings, unclassified, missing = audit(manifest)
    for finding in public_findings:
        print(f"FORBIDDEN {finding}")
    for finding in pending_findings:
        print(f"pending   {finding}")
    for rel in unclassified:
        print(f"UNCLASSIFIED {rel}: not in the manifest's public or pending list")
    for rel in missing:
        print(f"MISSING {rel}: listed in the manifest, absent from the tree")
    print(
        f"public files {len(manifest['public'])}, pending files {len(manifest['pending'])}, "
        f"public findings {len(public_findings)}, pending findings {len(pending_findings)}, "
        f"unclassified {len(unclassified)}, missing {len(missing)}"
    )
    failed = bool(public_findings or unclassified or missing or (args.strict and pending_findings))
    if failed:
        return 1
    if args.check:
        return 0
    files = list(manifest["public"])
    if args.with_pending:
        files += list(manifest["pending"])
    if args.with_dev:
        files += dev_files()
    if args.git_tree:
        print(write_git_tree(files))
        return 0
    if args.out.exists():
        print(f"{args.out} exists", file=sys.stderr)
        return 2
    export(files, args.out, allow_dev=args.with_dev)
    print(f"exported {len(files)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
