from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from . import compiled_topology_r5a2c_r2_data as topology_data
from . import compiled_topology_r5a2c_r2_executor as topology_executor
from . import transport_recolor_r5a0v2_model as v2_model
from . import transport_recolor_r5a0v2_oracles as v2_oracles
from .r5a2d_r1_corpus_resolver import PROJECT_ROOT, sha256_file
from .r5a2d_r2_candidate_ranking import stable_hash

AVAILABLE = "CANDIDATE_ENUMERATION_AVAILABLE"
UNAVAILABLE = "CANDIDATE_ENUMERATION_UNAVAILABLE"
INCONCLUSIVE = "CANDIDATE_ENUMERATION_INCONCLUSIVE"


def v2_catalog_payload() -> dict[str, Any]:
    return {
        "catalog_name": "transport_recolor_r5a0v2_oracles.SEALED_CLASSES",
        "candidate_ids": list(v2_oracles.SEALED_CLASSES),
        "raw_candidate_count": len(v2_oracles.SEALED_CLASSES),
    }


def topology_catalog_payload() -> dict[str, Any]:
    hypotheses = [hyp.to_json() for hyp in topology_data.candidate_hypotheses()]
    return {
        "catalog_name": "compiled_topology_r5a2c_r2_data.candidate_hypotheses()",
        "candidate_ids": [topology_candidate_id(row) for row in hypotheses],
        "raw_candidate_count": len(hypotheses),
        "hypotheses": hypotheses,
    }


def topology_candidate_id(payload: dict[str, Any]) -> str:
    marker = payload.get("marker_color_or_none")
    seed = payload.get("seed_condition_or_none")
    return (
        f"{payload['convention']}|passable={payload['passable_color']}|barrier={payload['barrier_color']}"
        f"|marker={marker}|seed={seed}|output={payload['output_recolor_color']}|adjacency={payload['adjacency']}"
    )


def candidate_surface_report() -> dict[str, Any]:
    v2_catalog = v2_catalog_payload()
    topology_catalog = topology_catalog_payload()
    v2_forward_parameters = list(inspect.signature(v2_model.SupportConditionedTransportRecolorR5A0V2.forward).parameters)
    topology_raw_parameters = list(inspect.signature(topology_executor.execute_from_raw_grids).parameters)
    topology_hypothesis_parameters = list(inspect.signature(topology_executor.execute_hypothesis).parameters)
    rows = [
        {
            "expert_name": "V2_transport_recolor",
            "frozen_source_hash": sha256_file(PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py"),
            "candidate_enumeration_available": True,
            "existing_candidate_catalog_name": v2_catalog["catalog_name"],
            "existing_candidate_catalog_hash": stable_hash(v2_catalog),
            "raw_candidate_count": v2_catalog["raw_candidate_count"],
            "candidate_execution_path": "read-only call to existing transport_recolor_r5a0v2_oracles.apply_sequence_to_grid(candidate_id)",
            "requires_forward_signature_change": False,
            "requires_new_runtime_argument": False,
            "requires_source_modification": False,
            "availability_verdict": AVAILABLE,
            "frozen_forward_parameters": v2_forward_parameters,
        },
        {
            "expert_name": "R5A-2C-R2_compiled_topology",
            "frozen_source_hash": sha256_file(PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_executor.py"),
            "candidate_enumeration_available": True,
            "existing_candidate_catalog_name": topology_catalog["catalog_name"],
            "existing_candidate_catalog_hash": stable_hash(topology_catalog),
            "raw_candidate_count": topology_catalog["raw_candidate_count"],
            "candidate_execution_path": "read-only enumeration of compiled_topology_r5a2c_r2_data.candidate_hypotheses() and existing execute_hypothesis(grid, hypothesis)",
            "requires_forward_signature_change": False,
            "requires_new_runtime_argument": False,
            "requires_source_modification": False,
            "availability_verdict": AVAILABLE,
            "frozen_execute_from_raw_grids_parameters": topology_raw_parameters,
            "frozen_execute_hypothesis_parameters": topology_hypothesis_parameters,
        },
    ]
    return {
        "audit_version": "R5A2D_R2_FROZEN_CANDIDATE_RECALL",
        "surfaces": rows,
        "all_available": all(row["availability_verdict"] == AVAILABLE for row in rows),
    }


def unavailable_surface_example() -> dict[str, Any]:
    return {
        "expert_name": "unit_unavailable",
        "candidate_enumeration_available": False,
        "existing_candidate_catalog_name": None,
        "existing_candidate_catalog_hash": None,
        "raw_candidate_count": 0,
        "candidate_execution_path": None,
        "requires_forward_signature_change": True,
        "requires_new_runtime_argument": True,
        "requires_source_modification": True,
        "availability_verdict": UNAVAILABLE,
    }
