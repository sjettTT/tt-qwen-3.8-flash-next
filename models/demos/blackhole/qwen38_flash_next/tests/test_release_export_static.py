"""The public export of this model directory: every public file is free of internal tokens, every file is classified,
and the public code imports nothing from ``tools/dev``.  Runs the exporter against the checked-out tree."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.tools.release import export_public_tree as release

MODEL_DIR = Path(__file__).resolve().parents[1]
PUBLIC_MODULES = (
    "chat.py",
    "checkpoint.py",
    "config.py",
    "reference.py",
    "tools/qwen38_chat_server.py",
    "tools/qwen38_chat_session.py",
    "tools/qwen38_sampling_step.py",
    "tools/hardware_profiles.py",
    "tools/resident_decode.py",
    "tools/evidence_records.py",
    "tools/qwen38_chat_cli.py",
    "tools/run_qwen38_chat_server.sh",
    "README.md",
    "tools/release/manifest.json",
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return release.load_manifest()


@pytest.fixture(scope="module")
def audit(manifest):
    return release.audit(manifest)


def test_manifest_classifies_every_file(manifest, audit) -> None:
    _public, _pending, unclassified, missing = audit
    assert unclassified == [], f"add to tools/release/manifest.json (public or pending): {unclassified}"
    assert missing == [], f"listed in the manifest, absent from the tree: {missing}"
    assert not set(manifest["public"]) & set(manifest["pending"])
    for rel in PUBLIC_MODULES:
        assert rel in manifest["public"], rel


def test_public_files_carry_no_internal_tokens(audit) -> None:
    public_findings, _pending, _unclassified, _missing = audit
    assert public_findings == [], "\n".join(str(finding) for finding in public_findings)


def test_pending_set_is_the_known_runtime_admission_seam(manifest, audit) -> None:
    _public, pending_findings, _unclassified, _missing = audit
    assert set(manifest["pending"]) == {
        "diagnostic_bf4.py",
        "tools/live_decode_diagnostic.py",
        "tools/runtime_admission.py",
    }
    classes = {finding.token_class for finding in pending_findings}
    assert classes <= {"user-path", "staging-root", "runtime-seal", "lab-hostname"}, classes
    assert pending_findings, "the pending files are clean: move them to the public list"


def test_forbidden_patterns_compile_and_catch_the_token_classes(manifest) -> None:
    patterns = {name: re.compile(pattern) for name, pattern in manifest["forbidden"].items()}
    # assembled so this file carries none of the tokens itself
    samples = {
        "lab-hostname": "ssh " + "f07" + "c" + "s04 and " + "c" + "s02",
        "user-path": "/home/" + "sje" + "tt/x and /home/" + "ttu" + "ser/y",
        "lab-ip": "172.27" + ".1.2 10.0" + ".0.17 100" + ".64.0.1",
        "staging-root": "$STA" + "GING/caches and qwen38-flash-" + "next-data",
        "device-lock": "exec 200>/run/" + "lock/tt-device-node-4.lock",
        "runtime-seal": "runtime-current-" + "main-f0b" + "17b2 and lib/_tt" + "nn.so",
        "board-identity": "boards={4: " + '"4E3FC211' + '49A3FEF0"}',
    }
    for name, sample in samples.items():
        assert patterns[name].search(sample), name
    clean = "tt-quietbox profile, TT_VISIBLE_DEVICES=0,1,2,3, --allocated-context 65536, mesh (1, 4)"
    assert not any(pattern.search(clean) for pattern in patterns.values())


def test_export_writes_the_public_tree_only(tmp_path, manifest) -> None:
    out = tmp_path / "public"
    assert release.main(["--out", str(out)]) == 0
    exported = sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file())
    assert exported == sorted(manifest["public"])
    assert not any(rel.startswith(release.DEV_DIRS) or release.is_dev_path(rel) for rel in exported)
    assert not any(rel in manifest["pending"] for rel in exported)
    for rel in ("tools/qwen38_chat_server.py", "tools/hardware_profiles.py"):
        assert (out / rel).read_bytes() == (MODEL_DIR / rel).read_bytes()


def test_export_with_pending_and_dev_adds_those_sets(tmp_path, manifest) -> None:
    out = tmp_path / "full"
    assert release.main(["--out", str(out), "--with-pending", "--with-dev"]) == 0
    exported = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    assert set(manifest["pending"]) <= exported
    assert set(release.dev_files()) <= exported


def test_export_refuses_an_existing_output(tmp_path) -> None:
    out = tmp_path / "taken"
    out.mkdir()
    assert release.main(["--out", str(out)]) == 2


def test_dev_paths_are_denied_everywhere(tmp_path, manifest) -> None:
    """A ``dev`` directory component fails the audit when listed, fails the copy when passed, and is what the
    tree walk skips; ``--with-dev`` lets only ``tools/dev`` and ``tests/dev`` through."""

    for rel in ("tools/dev/x.py", "tests/dev/test_x.py", "ttnn/dev/x.py", "tools/release/dev/x.json"):
        assert release.is_dev_path(rel), rel
    for rel in ("tools/x.py", "tools/devices.py", "dev.py", "tools/dev_profile.py", "tools/qb_dev_smoke.py"):
        assert not release.is_dev_path(rel), rel
    listed = dict(manifest)
    listed["public"] = [*manifest["public"], "tools/dev/x.py"]
    listed["pending"] = {**manifest["pending"], "ttnn/dev/y.py": "why"}
    public_findings, _pending, _unclassified, missing = release.audit(listed)
    assert [(finding.token_class, finding.path) for finding in public_findings] == [
        ("dev-path", "tools/dev/x.py"),
        ("dev-path", "ttnn/dev/y.py"),
    ]
    assert missing == []  # a denied path is reported once, as a finding
    with pytest.raises(ValueError, match="dev paths"):
        release.export(["chat.py", "tools/dev/x.py"], tmp_path / "a")
    with pytest.raises(ValueError, match="dev paths"):
        release.export(["ttnn/dev/x.py"], tmp_path / "b", allow_dev=True)
    assert not any(release.is_dev_path(rel) for rel in release.tree_files())
    assert all(rel.startswith(release.DEV_DIRS) for rel in release.dev_files())


def test_dev_import_rule_catches_every_spelling_and_leaves_device_paths_alone() -> None:
    model = "models.demos.blackhole.qwen38_flash_next"
    for text in (
        f"from {model}.tools.dev import resident_hybrid_smoke",
        f"import {model}.tools.dev.demo.production",
        "HERE/../tools/dev/run_x.sh",
        f'Path("{model.replace(".", "/")}/tests/dev/test_x.py")',
        'root / "tools" / "dev" / "demo" / "production.py"',
    ):
        assert release.DEV_IMPORT.search(text), text
    for text in (
        'Path("/dev/tenstorrent/by-id/blackhole-0")',
        "TT_VISIBLE_DEVICES=0,1,2,3 tools/hardware_profiles.py",
        "the developer's device nodes",
        "python -m pytest tests",
    ):
        assert not release.DEV_IMPORT.search(text), text


def test_public_modules_import_without_loading_dev_code() -> None:
    """The runtime form of the import guard: a fresh interpreter imports every public Python module and nothing under
    ``tools/dev`` or ``tests/dev`` enters ``sys.modules``."""

    pytest.importorskip("ttnn")
    package = "models.demos.blackhole.qwen38_flash_next"
    modules = [
        f"{package}.{rel[:-3].replace('/', '.')}"
        for rel in release.load_manifest()["public"]
        if rel.endswith(".py") and not rel.startswith("tests/") and not rel.endswith("__init__.py")
    ]
    # the pending admission modules are imported by the server: they are public code with pinned identities
    pending = ("diagnostic_bf4", "tools.live_decode_diagnostic", "tools.runtime_admission")
    modules += [f"{package}.{name}" for name in pending]
    script = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "loaded = sorted(name for name in sys.modules if '.tools.dev' in name or '.tests.dev' in name)\n"
        "print(json.dumps(loaded))\n"
    )
    python_path = os.pathsep.join(filter(None, (str(release.REPO_ROOT), os.environ.get("PYTHONPATH"))))
    run = subprocess.run(
        [sys.executable, "-c", script],
        cwd=release.REPO_ROOT,
        env={**os.environ, "PYTHONPATH": python_path},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    assert json.loads(run.stdout.strip().splitlines()[-1]) == []


def test_manifest_is_sorted_and_stable() -> None:
    document = json.loads((MODEL_DIR / "tools/release/manifest.json").read_text())
    assert document["public"] == sorted(document["public"])
    assert list(document) == ["public", "pending", "forbidden"]
