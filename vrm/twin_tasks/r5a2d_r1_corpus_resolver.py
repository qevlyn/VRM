from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path("/mnt/d/VRM")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage"
CORPUS_MANIFEST_PATH = ARTIFACT_ROOT / "corpus_manifest_precommit.json"
AUDIT_VERSION = "R5A2D_R1_FROZEN_LIBRARY_ARC_NATIVE_COVERAGE_AUDIT"
NO_LOCAL_CORPUS_VERDICT = "R5A2D_R1_FROZEN_LIBRARY_COVERAGE_INCONCLUSIVE_NO_LOCAL_ARC_CORPUS"


class NoLocalArcCorpusError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusSelection:
    corpus_name: str
    corpus_root: Path
    corpus_root_relative_or_configured_path: str
    provenance_statement: str
    resolution_rule: str
    task_files: tuple[dict[str, Any], ...]
    corpus_manifest_sha256: str
    manifest_path: Path
    frozen_from_existing_manifest: bool


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _is_probable_arc_agi2_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {item.name for item in path.glob("*.json")}
    return any(name.endswith("_challenges.json") for name in names) and "arc-agi" in path.name.lower() and "2" in path.name


def _is_probable_arc_agi1_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {item.name for item in path.glob("*.json")}
    has_split_files = any("training" in name.lower() and name.endswith(".json") for name in names)
    return has_split_files and "arc" in path.name.lower() and "2" not in path.name


def _is_configured_arc_style_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    json_files = sorted(path.glob("*.json"))
    if any(file.name.endswith("_challenges.json") for file in json_files):
        return True
    return any(file.name != "sample_submission.json" for file in json_files)


def _candidate_json_files(root: Path) -> tuple[Path, ...]:
    direct = sorted(path for path in root.glob("*.json") if path.name != "sample_submission.json")
    if any(path.name.endswith("_challenges.json") for path in direct):
        return tuple(
            path
            for path in direct
            if path.name.endswith("_challenges.json") or path.name.endswith("_solutions.json")
        )
    return tuple(direct)


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _search_roots(project_root: Path, predicate: Any) -> list[Path]:
    ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
    roots: list[Path] = []
    for current, dirnames, _ in os.walk(project_root):
        dirnames[:] = sorted(name for name in dirnames if name not in ignored)
        path = Path(current)
        if predicate(path):
            roots.append(path)
    return sorted(roots, key=lambda item: str(item))


def _make_manifest_payload(
    *,
    project_root: Path,
    corpus_root: Path,
    corpus_name: str,
    provenance_statement: str,
    resolution_rule: str,
    configured_path: bool,
) -> dict[str, Any]:
    files = []
    for path in _candidate_json_files(corpus_root):
        stat = path.stat()
        files.append(
            {
                "relative_path": _relative_to_project(path, project_root),
                "corpus_relative_path": str(path.relative_to(corpus_root)),
                "byte_size": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    body = {
        "audit_version": AUDIT_VERSION,
        "corpus_name": corpus_name,
        "corpus_root_relative_or_configured_path": str(corpus_root) if configured_path else _relative_to_project(corpus_root, project_root),
        "corpus_root_absolute_path": str(corpus_root),
        "provenance_statement": provenance_statement,
        "resolution_rule": resolution_rule,
        "task_file_count": len(files),
        "task_files": files,
        "deterministic_task_file_order": True,
        "resolver_grid_value_inspection": False,
        "manifest_written_before_task_grid_interpretation": True,
    }
    body["corpus_manifest_sha256"] = stable_hash({key: value for key, value in body.items() if key != "corpus_manifest_sha256"})
    return body


def _selection_from_manifest(path: Path, project_root: Path) -> CorpusSelection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CorpusSelection(
        corpus_name=payload["corpus_name"],
        corpus_root=Path(payload["corpus_root_absolute_path"]),
        corpus_root_relative_or_configured_path=payload["corpus_root_relative_or_configured_path"],
        provenance_statement=payload["provenance_statement"],
        resolution_rule=payload["resolution_rule"],
        task_files=tuple(payload["task_files"]),
        corpus_manifest_sha256=payload["corpus_manifest_sha256"],
        manifest_path=path,
        frozen_from_existing_manifest=True,
    )


def resolve_and_precommit_corpus(
    *,
    project_root: Path = PROJECT_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
    environ: Mapping[str, str] | None = None,
) -> CorpusSelection:
    env = os.environ if environ is None else environ
    manifest_path = artifact_root / "corpus_manifest_precommit.json"
    if manifest_path.exists():
        return _selection_from_manifest(manifest_path, project_root)

    configured = env.get("VRM_ARC_COVERAGE_CORPUS")
    if configured:
        root = Path(configured)
        if root.exists() and _is_configured_arc_style_root(root):
            payload = _make_manifest_payload(
                project_root=project_root,
                corpus_root=root,
                corpus_name=root.name,
                provenance_statement="local ARC-style corpus; provenance not independently verified by this audit",
                resolution_rule="VRM_ARC_COVERAGE_CORPUS",
                configured_path=True,
            )
            write_json(manifest_path, payload)
            return _selection_from_manifest(manifest_path, project_root)

    arc2 = _search_roots(project_root, _is_probable_arc_agi2_root)
    if arc2:
        root = arc2[0]
        payload = _make_manifest_payload(
            project_root=project_root,
            corpus_root=root,
            corpus_name=root.name,
            provenance_statement="local ARC-style corpus; provenance not independently verified by this audit",
            resolution_rule="first_valid_local_arc_agi_2_sorted_path_search",
            configured_path=False,
        )
        write_json(manifest_path, payload)
        return _selection_from_manifest(manifest_path, project_root)

    arc1 = _search_roots(project_root, _is_probable_arc_agi1_root)
    if arc1:
        root = arc1[0]
        payload = _make_manifest_payload(
            project_root=project_root,
            corpus_root=root,
            corpus_name=root.name,
            provenance_statement="local ARC-style corpus; provenance not independently verified by this audit",
            resolution_rule="first_valid_local_arc_agi_1_sorted_path_search",
            configured_path=False,
        )
        write_json(manifest_path, payload)
        return _selection_from_manifest(manifest_path, project_root)

    raise NoLocalArcCorpusError(NO_LOCAL_CORPUS_VERDICT)
