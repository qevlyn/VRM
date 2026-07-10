from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
import tokenize
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import compiled_topology_r5a2c_r2_checker as checker
from . import compiled_topology_r5a2c_r2_data as data
from . import compiled_topology_r5a2c_r2_executor as executor

PROJECT_ROOT = Path("/mnt/d/VRM")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a2c_r2_topology"
CLOSURE_DOMAIN_PATH = ARTIFACT_ROOT / "closure_domain.json"
SHORTCUT_CATALOG_PATH = ARTIFACT_ROOT / "shortcut_catalog.json"
GENERATOR_AUDIT_PATH = ARTIFACT_ROOT / "generator_audit.json"
SHORTCUT_AUDIT_PATH = ARTIFACT_ROOT / "shortcut_audit.json"
EVALUATION_REPORT_PATH = ARTIFACT_ROOT / "evaluation_report.json"
ABLATION_REPORT_PATH = ARTIFACT_ROOT / "ablation_report.json"
FINAL_REPORT_PATH = ARTIFACT_ROOT / "final_report.json"
PYTEST_RESULT_PATH = ARTIFACT_ROOT / "pytest_result.txt"

AUDIT_VERSION = "R5A2C_R2_COMPILED_TOPOLOGY_AUDIT_V1"
PASS_VERDICT = "R5A2C_R2_COMPILED_TOPOLOGY_PRIMITIVE_PASS"
PARTIAL_VERDICT = "R5A2C_R2_COMPILED_TOPOLOGY_PRIMITIVE_PARTIAL"
GENERATOR_INVALID_VERDICT = "R5A2C_R2_COMPILED_TOPOLOGY_GENERATOR_INVALID"
INCONCLUSIVE_VERDICT = "R5A2C_R2_COMPILED_TOPOLOGY_INCONCLUSIVE"

MANDATORY_NON_CLAIM = """This audit measures only whether one additional narrow compiled specialist can
be implemented and validated under a declared finite topology hypothesis language.

It does not measure:
- the human design/search cost of discovering that hypothesis language;
- coverage of the language on real ARC tasks;
- routing among a large specialist library;
- search cost when a future library becomes large;
- ARC generality;
- zero-compute ARC solving;
- hidden-tool emergence."""

PROTECTED_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a2a_frontier_audit.py",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/learning_dependency_audit_report.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a2a_frontier/final_report.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_report_hash(report: dict[str, Any]) -> str:
    stable = copy.deepcopy(report)
    stable.pop("created_utc", None)
    stable.pop("report_body_sha256_excluding_timestamp", None)
    stable.pop("generated_artifact_hashes", None)
    return stable_hash(stable)


def protected_hashes() -> dict[str, str | None]:
    return {str(path): sha256_file(path) if path.exists() else None for path in PROTECTED_PATHS}


def source_without_strings_and_comments(source: str) -> str:
    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        pieces.append(" " if token.type in {tokenize.STRING, tokenize.COMMENT} else token.string)
    return " ".join(pieces)


def parameter_free_report() -> dict[str, Any]:
    source_paths = [
        PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_data.py",
        PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_checker.py",
        PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_executor.py",
        PROJECT_ROOT / "vrm/twin_tasks/run_compiled_topology_r5a2c_r2_audit.py",
    ]
    stripped = "\n".join(source_without_strings_and_comments(path.read_text(encoding="utf-8")) for path in source_paths)
    forbidden = [term for term in ("nn.Parameter", "torch.optim", ".backward(", "optimizer.step", "torch.load", "torch.save") if term in stripped]
    return {
        "no_nn_module_with_trainable_parameters_exists": "nn.Module" not in stripped,
        "no_nn_parameter_exists": "nn.Parameter" not in stripped,
        "no_optimizer_constructed": "torch.optim" not in stripped and "optimizer" not in stripped,
        "no_backward_called": ".backward(" not in stripped,
        "no_checkpoint_path_read": "torch.load" not in stripped,
        "no_checkpoint_path_written": "torch.save" not in stripped,
        "forbidden_code_terms_found": forbidden,
        "passed": not forbidden and "nn.Module" not in stripped and "optimizer" not in stripped,
    }


def checker_independence_report() -> dict[str, Any]:
    source = (PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_checker.py").read_text(encoding="utf-8")
    forbidden = [
        term
        for term in (
            "compiled_topology_r5a2c_r2_executor",
            "transport_recolor_r5a0v2",
            "SupportConditionedTransportRecolorR5A0V2",
        )
        if term in source
    ]
    return {"forbidden_import_or_call_terms": forbidden, "passed": not forbidden}


def parse_pytest_result(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "passed": None, "summary": None}
    if not path.exists():
        return {"provided": True, "passed": False, "summary": "missing pytest result", "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = ""
    for line in reversed(text.splitlines()):
        if " passed" in line or " failed" in line:
            summary = line.strip()
            break
    return {
        "provided": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "summary": summary,
        "passed": " passed" in text and " failed" not in text and " error" not in text.lower(),
    }


def all_accepted_episodes(suites: dict[str, list[data.TopologyEpisode]]) -> list[data.TopologyEpisode]:
    return [episode for episodes in suites.values() for episode in episodes]


def support_inputs_outputs(episode: data.TopologyEpisode) -> tuple[list[tuple[tuple[int, ...], ...]], list[tuple[tuple[int, ...], ...]]]:
    return [pair.input for pair in episode.supports], [pair.output for pair in episode.supports]


def evaluate_episode(episode: data.TopologyEpisode, *, ablation: str | None = None) -> dict[str, Any]:
    support_inputs, support_outputs = support_inputs_outputs(episode)
    result = executor.execute_from_raw_grids(support_inputs, support_outputs, episode.query_input_grid, ablation=ablation)
    output = result.get("output_grid")
    return {
        "episode_id": episode.episode_id,
        "suite_name": episode.suite_name,
        "convention": episode.convention,
        "status": result["status"],
        "exact": output == episode.query_output_grid,
        "output_grid": data.grid_to_list(output) if output is not None else None,
        "selected_hypothesis": result.get("selected_hypothesis"),
        "consistent_hypotheses": result.get("consistent_hypotheses", []),
    }


def summarize_exact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_convention = defaultdict(lambda: {"count": 0, "exact": 0})
    by_suite = defaultdict(lambda: {"count": 0, "exact": 0})
    for row in rows:
        by_convention[row["convention"]]["count"] += 1
        by_convention[row["convention"]]["exact"] += int(row["exact"])
        by_suite[row["suite_name"]]["count"] += 1
        by_suite[row["suite_name"]]["exact"] += int(row["exact"])
    return {
        "episode_count": len(rows),
        "overall_exact": sum(int(row["exact"]) for row in rows) / max(len(rows), 1),
        "exact_by_convention": {key: value["exact"] / value["count"] for key, value in sorted(by_convention.items())},
        "exact_by_suite": {key: value["exact"] / value["count"] for key, value in sorted(by_suite.items())},
        "first_failure": next((row for row in rows if not row["exact"]), None),
    }


def evaluation_report(suites: dict[str, list[data.TopologyEpisode]]) -> dict[str, Any]:
    rows = [evaluate_episode(episode) for episode in all_accepted_episodes(suites)]
    report = {"audit_version": AUDIT_VERSION, "rows": rows, "summary": summarize_exact(rows)}
    report["randomized_suite_exact"] = report["summary"]["exact_by_suite"].get("randomized_suite")
    report["generator_stress_suite_exact"] = report["summary"]["exact_by_suite"].get("generator_stress_suite")
    report["generator_stress_interpretation"] = "generator/domain stress evidence only; not learner-style held-out generalization evidence."
    return report


def candidate_components(grid: tuple[tuple[int, ...], ...], passable_color: int) -> list[set[tuple[int, int]]]:
    return data.connected_components(data.passable_positions(grid, passable_color))


def infer_roles(episode: data.TopologyEpisode) -> data.TopologyHypothesis:
    changed = []
    for pair in episode.supports:
        for y in range(data.GRID_SIZE):
            for x in range(data.GRID_SIZE):
                if pair.input[y][x] != pair.output[y][x]:
                    changed.append((pair.input[y][x], pair.output[y][x]))
    if changed:
        passable = Counter(item[0] for item in changed).most_common(1)[0][0]
        output = Counter(item[1] for item in changed).most_common(1)[0][0]
    else:
        passable, output = data.PASSABLE_COLORS[0], data.OUTPUT_COLORS[0]
    markers = [value for row in episode.query_input_grid for value in row if value in data.MARKER_COLORS]
    marker = Counter(markers).most_common(1)[0][0] if markers else None
    return data.TopologyHypothesis("seed_reachable_recolor", passable, data.BARRIER_COLORS[0], marker, data.SEED_CONDITIONS[0] if marker else None, output)


def component_predicate_values(episode: data.TopologyEpisode, component: set[tuple[int, int]], components: list[set[tuple[int, int]]]) -> dict[str, bool]:
    sizes = [len(item) for item in components]
    median_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
    hyp = infer_roles(episode)
    barrier_density = data.grid_metrics(episode.query_input_grid, hyp)["barrier_density"]
    seed_distance = data.shortest_distance_to_border(component)
    seed_distances = [data.shortest_distance_to_border(comp) or 0 for comp in components]
    median_distance = sorted(seed_distances)[len(seed_distances) // 2] if seed_distances else 0
    markers = data.marker_positions(episode.query_input_grid, hyp.marker_color_or_none)
    marker_adjacent = any(neighbor in component for marker in markers for neighbor in data.neighbors(marker))
    touches_border = any(x in {0, data.GRID_SIZE - 1} or y in {0, data.GRID_SIZE - 1} for x, y in component)
    return {
        "is_largest_component": len(component) == max(sizes) if sizes else False,
        "is_smallest_component": len(component) == min(sizes) if sizes else False,
        "touches_border": touches_border,
        "does_not_touch_border": not touches_border,
        "is_marker_adjacent": marker_adjacent,
        "is_not_marker_adjacent": not marker_adjacent,
        "has_region_size_above_median": len(component) > median_size,
        "has_region_size_below_median": len(component) < median_size,
        "has_barrier_density_above_median": barrier_density > barrier_density,
        "has_seed_distance_above_median": (seed_distance or 0) > median_distance,
    }


def recolor_component(grid: tuple[tuple[int, ...], ...], component: set[tuple[int, int]], passable: int, output: int) -> tuple[tuple[int, ...], ...]:
    out = [list(row) for row in grid]
    for x, y in component:
        if grid[y][x] == passable:
            out[y][x] = output
    return data.freeze_grid(out)


def shortcut_prediction(episode: data.TopologyEpisode, rule: dict[str, Any]) -> tuple[tuple[int, ...], ...] | None:
    hyp = infer_roles(episode)
    name = rule["name"]
    if name == "identity_output":
        return episode.query_input_grid
    if name == "paint_all_passable":
        out = [list(row) for row in episode.query_input_grid]
        for x, y in data.passable_positions(episode.query_input_grid, hyp.passable_color):
            out[y][x] = hyp.output_recolor_color
        return data.freeze_grid(out)
    if name == "always_seed_reachable":
        return data.apply_hypothesis_reference(episode.query_input_grid, hyp)
    if name == "always_border_reachable":
        return data.apply_hypothesis_reference(episode.query_input_grid, data.TopologyHypothesis("border_reachable_recolor", hyp.passable_color, hyp.barrier_color, None, None, hyp.output_recolor_color))
    if name == "always_enclosed":
        return data.apply_hypothesis_reference(episode.query_input_grid, data.TopologyHypothesis("enclosed_region_recolor", hyp.passable_color, hyp.barrier_color, None, None, hyp.output_recolor_color))
    if name == "one_step_only":
        return data.apply_hypothesis_reference(episode.query_input_grid, episode.hypothesis, one_step_only=True)
    if name == "barrier_ignored":
        return data.apply_hypothesis_reference(episode.query_input_grid, episode.hypothesis, barrier_ignored=True)
    if name == "wrong_adjacency_8_connected":
        return data.apply_hypothesis_reference(episode.query_input_grid, episode.hypothesis, adjacency="8_connected")
    comps = candidate_components(episode.query_input_grid, hyp.passable_color)
    if not comps:
        return None
    selected: set[tuple[int, int]] | None = None
    if name == "always_largest_passable_component":
        selected = max(comps, key=lambda comp: (len(comp), sorted(comp)))
    elif name == "always_non_border_component":
        candidates = [comp for comp in comps if not any(x in {0, data.GRID_SIZE - 1} or y in {0, data.GRID_SIZE - 1} for x, y in comp)]
        selected = candidates[0] if len(candidates) == 1 else None
    elif name == "always_marker_adjacent_component":
        markers = data.marker_positions(episode.query_input_grid, hyp.marker_color_or_none)
        candidates = [comp for comp in comps if any(neighbor in comp for marker in markers for neighbor in data.neighbors(marker))]
        selected = candidates[0] if len(candidates) == 1 else None
    elif rule["kind"] in {"pairwise_conjunction", "ordered_lexicographic"}:
        rows = [(comp, component_predicate_values(episode, comp, comps)) for comp in comps]
        if rule["kind"] == "pairwise_conjunction":
            candidates = [comp for comp, values in rows if all(values[predicate] for predicate in rule["predicates"])]
        else:
            candidates = [comp for comp, values in rows if values[rule["predicates"][0]]]
            if len(candidates) > 1:
                values_by_comp = {tuple(sorted(comp)): vals for comp, vals in rows}
                candidates = [comp for comp in candidates if values_by_comp[tuple(sorted(comp))][rule["predicates"][1]]]
        selected = candidates[0] if len(candidates) == 1 else None
    if selected is None:
        return None
    return recolor_component(episode.query_input_grid, selected, hyp.passable_color, hyp.output_recolor_color)


def shortcut_audit(episodes: list[data.TopologyEpisode], catalog: dict[str, Any]) -> dict[str, Any]:
    rows = []
    by_convention: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in catalog["rules"]:
        exact = 0
        count = 0
        convention_counts = defaultdict(lambda: {"count": 0, "exact": 0})
        for episode in episodes:
            pred = shortcut_prediction(episode, rule)
            ok = pred == episode.query_output_grid
            exact += int(ok)
            count += 1
            convention_counts[episode.convention]["count"] += 1
            convention_counts[episode.convention]["exact"] += int(ok)
        row = {
            "rule_name": rule["name"],
            "kind": rule["kind"],
            "overall_exact": exact / max(count, 1),
            "exact_by_convention": {key: value["exact"] / value["count"] for key, value in sorted(convention_counts.items())},
        }
        rows.append(row)
        for convention, stats in convention_counts.items():
            by_convention[convention].append({"rule_name": rule["name"], "kind": rule["kind"], "exact": stats["exact"] / stats["count"]})
    best = max(rows, key=lambda row: (row["overall_exact"], row["rule_name"]))
    def best_of(kind: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [row for row in subset if row["kind"] == kind]
        return max(candidates, key=lambda row: (row.get("overall_exact", row.get("exact", 0.0)), row["rule_name"]))
    return {
        "shortcut_catalog_sha256": catalog["shortcut_catalog_sha256"],
        "single_rule_count": catalog["single_rule_count"],
        "pairwise_conjunction_count": catalog["pairwise_conjunction_count"],
        "lexicographic_rule_count": catalog["lexicographic_rule_count"],
        "total_rule_count": catalog["total_rule_count"],
        "best_single_rule_exact": best_of("single", rows)["overall_exact"],
        "best_conjunctive_rule_exact": best_of("pairwise_conjunction", rows)["overall_exact"],
        "best_lexicographic_rule_exact": best_of("ordered_lexicographic", rows)["overall_exact"],
        "best_shortcut_exact": best["overall_exact"],
        "best_shortcut_name": best["rule_name"],
        "passes_threshold": best["overall_exact"] <= data.SHORTCUT_THRESHOLD,
        "by_convention": {
            convention: {
                "best_single_rule_exact": best_of("single", values)["exact"],
                "best_conjunctive_rule_exact": best_of("pairwise_conjunction", values)["exact"],
                "best_lexicographic_rule_exact": best_of("ordered_lexicographic", values)["exact"],
                "best_shortcut_exact": max(values, key=lambda row: (row["exact"], row["rule_name"]))["exact"],
                "best_shortcut_name": max(values, key=lambda row: (row["exact"], row["rule_name"]))["rule_name"],
            }
            for convention, values in sorted(by_convention.items())
        },
        "rows": rows,
    }


def generator_audit(suites: dict[str, list[data.TopologyEpisode]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = all_accepted_episodes(suites)
    checker_rows = [{"episode_id": episode.episode_id, **checker.verify_episode(episode)} for episode in accepted]
    consistent_counts = Counter(str(row["consistent_hypothesis_count"]) for row in checker_rows)
    accepted_metrics = [episode.metadata["query_metrics"] for episode in accepted]
    rejected_metrics = [row["metrics"] for row in rejected]
    all_generated = len(accepted) + len(rejected)
    multiple_rejected = sum(1 for row in rejected if row["rejection_reason"] == "multiple_consistent_hypotheses")
    activity_rate = multiple_rejected / max(all_generated, 1)
    if activity_rate >= 0.05:
        activity_label = "IDENTIFIABILITY_GATE_ACTIVE"
    elif activity_rate > 0.0:
        activity_label = "IDENTIFIABILITY_GATE_LOW_ACTIVITY"
    else:
        activity_label = "IDENTIFIABILITY_GATE_VACUOUS"
    return {
        "all_generated_episode_count": all_generated,
        "accepted_episode_count": len(accepted),
        "rejected_episode_count": len(rejected),
        "rejection_reason_counts": dict(Counter(row["rejection_reason"] for row in rejected)),
        "candidate_hypothesis_count_distribution": {str(len(data.candidate_hypotheses())): all_generated},
        "consistent_hypothesis_count_before_rejection_distribution": dict(consistent_counts | Counter({"multiple": multiple_rejected})),
        "identifiability_gate_activity_rate": activity_rate,
        "identifiability_gate_activity_label": activity_label,
        "checker_episode_rows": checker_rows,
        "complete_tuple_uniqueness": {
            "accepted_with_exactly_one_complete_tuple": sum(1 for row in checker_rows if row["complete_tuple_unique"]),
            "accepted_episode_count": len(accepted),
            "all_accepted_unique": all(row["complete_tuple_unique"] for row in checker_rows),
        },
        "accepted_geometry_distributions": data.distribution_report(accepted_metrics),
        "rejected_geometry_distributions": data.distribution_report(rejected_metrics),
        "accepted_vs_rejected_selection_effect_statement": "Rejected episodes are intentionally empty ambiguous controls; the gate selects valid nonempty topology instances rather than a geometrically easier subset of valid topology tasks.",
        "counterbalancing_summary": data.counterbalance_report(accepted),
        "domain_containment_all_accepted": all(row["domain_containment"] for row in checker_rows),
    }


def ablation_report(suites: dict[str, list[data.TopologyEpisode]]) -> dict[str, Any]:
    episodes = all_accepted_episodes(suites)
    normal_rows = [evaluate_episode(episode) for episode in episodes]
    normal = summarize_exact(normal_rows)
    ablations = {}
    for ablation in ("no_propagation", "one_step_only", "wrong_adjacency_8_connected", "barrier_ignored", "border_seed_disabled", "support_consistency_disabled"):
        rows = [evaluate_episode(episode, ablation=ablation) for episode in episodes]
        summary = summarize_exact(rows)
        drops = {
            convention: normal["exact_by_convention"].get(convention, 0.0) - summary["exact_by_convention"].get(convention, 0.0)
            for convention in data.CONVENTIONS
        }
        ablations[ablation] = {
            "exact_by_convention": summary["exact_by_convention"],
            "randomized_suite_exact": summary["exact_by_suite"].get("randomized_suite"),
            "generator_stress_suite_exact": summary["exact_by_suite"].get("generator_stress_suite"),
            "absolute_drop_vs_normal": drops,
            "first_counterexample": next((row for row in rows if not row["exact"]), None),
            "passes_relevant_drop_threshold": max(drops.values()) >= 0.30,
        }
    adjacency_episodes = [episode for episode in episodes if episode.metadata["adjacency_discrimination"] and episode.suite_name in {"randomized_suite", "generator_stress_suite"}]
    normal_adj = [evaluate_episode(episode) for episode in adjacency_episodes]
    wrong_adj = [evaluate_episode(episode, ablation="wrong_adjacency_8_connected") for episode in adjacency_episodes]
    normal_exact = summarize_exact(normal_adj)["overall_exact"]
    wrong_exact = summarize_exact(wrong_adj)["overall_exact"]
    return {
        "normal_exact_by_convention": normal["exact_by_convention"],
        "normal_overall_exact": normal["overall_exact"],
        "ablations": ablations,
        "all_relevant_ablations_pass_drop_threshold": all(row["passes_relevant_drop_threshold"] for row in ablations.values()),
        "adjacency_discrimination": {
            "adjacency_discrimination_episode_count": len(adjacency_episodes),
            "adjacency_discrimination_episode_fraction": len(adjacency_episodes) / max(len([episode for episode in episodes if episode.suite_name in {"randomized_suite", "generator_stress_suite"}]), 1),
            "normal_4_connected_exact_on_adjacency_slice": normal_exact,
            "wrong_8_connected_exact_on_adjacency_slice": wrong_exact,
            "absolute_drop_on_adjacency_slice": normal_exact - wrong_exact,
            "first_adjacency_counterexample": next((row for row in wrong_adj if not row["exact"]), None),
            "passes": (normal_exact - wrong_exact) >= 0.30,
        },
    }


def build_report(pytest_result: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protected_before = protected_hashes()
    closure_domain = data.closure_domain_payload()
    suites, rejected = data.generate_suites()
    accepted = all_accepted_episodes(suites)
    shortcut_catalog = data.enumerate_shortcut_catalog()
    generator = generator_audit(suites, rejected)
    shortcuts = shortcut_audit(accepted, shortcut_catalog)
    evaluation = evaluation_report(suites)
    ablations = ablation_report(suites)
    protected_after = protected_hashes()
    pfree = parameter_free_report()
    golden = checker.verify_golden_cases()
    independence = checker_independence_report()
    hard_gates = {
        "parameter_free_no_training": pfree["passed"],
        "checker_independent": independence["passed"],
        "golden_cases_pass": golden["passed"],
        "complete_tuple_uniqueness": generator["complete_tuple_uniqueness"]["all_accepted_unique"],
        "randomized_suite_exact_ge_0_99": evaluation["randomized_suite_exact"] >= 0.99,
        "generator_stress_suite_exact_ge_0_99": evaluation["generator_stress_suite_exact"] >= 0.99,
        "shortcut_ceiling_not_exceeded": shortcuts["passes_threshold"],
        "relevant_ablations_collapse": ablations["all_relevant_ablations_pass_drop_threshold"],
        "adjacency_discrimination_coverage_ge_25_percent": ablations["adjacency_discrimination"]["adjacency_discrimination_episode_fraction"] >= 0.25,
        "adjacency_discrimination_drop_ge_0_30": ablations["adjacency_discrimination"]["passes"],
        "domain_containment": generator["domain_containment_all_accepted"],
        "protected_hashes_unchanged": protected_before == protected_after,
        "pytest_passed_if_provided": parse_pytest_result(pytest_result)["passed"] is not False,
    }
    if not pfree["passed"] or not independence["passed"] or protected_before != protected_after:
        verdict = INCONCLUSIVE_VERDICT
    elif not shortcuts["passes_threshold"]:
        verdict = GENERATOR_INVALID_VERDICT
    elif all(hard_gates.values()):
        verdict = PASS_VERDICT
    elif pfree["passed"] and golden["passed"] and generator["complete_tuple_uniqueness"]["all_accepted_unique"]:
        verdict = PARTIAL_VERDICT
    else:
        verdict = INCONCLUSIVE_VERDICT
    final = {
        "audit_version": AUDIT_VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": verdict,
        "mandatory_non_claim": MANDATORY_NON_CLAIM,
        "scope_boundary": "These are one reachability primitive exercised under three support-conditioned conventions. They are not three independent primitive discoveries.",
        "top_level_mechanism_finding": {
            "normal_executor_exact": evaluation["summary"]["overall_exact"],
            "shortcut_ceiling_not_exceeded": shortcuts["passes_threshold"],
            "relevant_ablation_collapse": ablations["all_relevant_ablations_pass_drop_threshold"],
            "adjacency_discrimination_collapse": ablations["adjacency_discrimination"]["passes"],
            "domain_and_identifiability_gates_pass": generator["domain_containment_all_accepted"] and generator["complete_tuple_uniqueness"]["all_accepted_unique"],
        },
        "closure_domain_sha256": closure_domain["closure_domain_sha256"],
        "shortcut_catalog_sha256": shortcut_catalog["shortcut_catalog_sha256"],
        "parameter_free_no_training_evidence": pfree,
        "checker_independence_report": independence,
        "golden_case_report": golden,
        "generator_summary": {
            "all_generated_episode_count": generator["all_generated_episode_count"],
            "accepted_episode_count": generator["accepted_episode_count"],
            "rejected_episode_count": generator["rejected_episode_count"],
            "rejection_reason_counts": generator["rejection_reason_counts"],
            "identifiability_gate_activity_rate": generator["identifiability_gate_activity_rate"],
            "identifiability_gate_activity_label": generator["identifiability_gate_activity_label"],
        },
        "accepted_vs_rejected_geometry_distributions": {
            "accepted": generator["accepted_geometry_distributions"],
            "rejected": generator["rejected_geometry_distributions"],
            "selection_effect_statement": generator["accepted_vs_rejected_selection_effect_statement"],
        },
        "counterbalancing_summary": generator["counterbalancing_summary"],
        "complete_tuple_uniqueness_results": generator["complete_tuple_uniqueness"],
        "normal_executor_exact_by_convention": evaluation["summary"]["exact_by_convention"],
        "randomized_suite_result": evaluation["randomized_suite_exact"],
        "generator_stress_result": {
            "exact": evaluation["generator_stress_suite_exact"],
            "interpretation": evaluation["generator_stress_interpretation"],
        },
        "shortcut_summary": {
            "single_rule_count": shortcuts["single_rule_count"],
            "pairwise_conjunction_count": shortcuts["pairwise_conjunction_count"],
            "lexicographic_rule_count": shortcuts["lexicographic_rule_count"],
            "total_rule_count": shortcuts["total_rule_count"],
            "best_shortcut_exact": shortcuts["best_shortcut_exact"],
            "best_shortcut_name": shortcuts["best_shortcut_name"],
            "best_single_rule_exact": shortcuts["best_single_rule_exact"],
            "best_conjunctive_rule_exact": shortcuts["best_conjunctive_rule_exact"],
            "best_lexicographic_rule_exact": shortcuts["best_lexicographic_rule_exact"],
        },
        "ablation_summary": ablations,
        "pytest_result": parse_pytest_result(pytest_result),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "hard_gates": hard_gates,
        "stable_two_run_hash_comparison_required": True,
        "generated_artifact_hashes": {},
        "narrowest_justified_next_action": "stop; do not automatically build another compiled specialist or outer learner",
    }
    final["report_body_sha256_excluding_timestamp"] = stable_report_hash(final)
    return final, closure_domain, shortcut_catalog, generator, shortcuts, evaluation, ablations


def write_artifacts(final: dict[str, Any], closure_domain: dict[str, Any], shortcut_catalog: dict[str, Any], generator: dict[str, Any], shortcuts: dict[str, Any], evaluation: dict[str, Any], ablations: dict[str, Any]) -> None:
    write_json(CLOSURE_DOMAIN_PATH, closure_domain)
    write_json(SHORTCUT_CATALOG_PATH, shortcut_catalog)
    write_json(GENERATOR_AUDIT_PATH, generator)
    write_json(SHORTCUT_AUDIT_PATH, shortcuts)
    write_json(EVALUATION_REPORT_PATH, evaluation)
    write_json(ABLATION_REPORT_PATH, ablations)
    final_copy = copy.deepcopy(final)
    final_copy["generated_artifact_hashes"] = {
        "closure_domain.json": sha256_file(CLOSURE_DOMAIN_PATH),
        "shortcut_catalog.json": sha256_file(SHORTCUT_CATALOG_PATH),
        "generator_audit.json": sha256_file(GENERATOR_AUDIT_PATH),
        "shortcut_audit.json": sha256_file(SHORTCUT_AUDIT_PATH),
        "evaluation_report.json": sha256_file(EVALUATION_REPORT_PATH),
        "ablation_report.json": sha256_file(ABLATION_REPORT_PATH),
    }
    final_copy["report_body_sha256_excluding_timestamp"] = stable_report_hash(final_copy)
    write_json(FINAL_REPORT_PATH, final_copy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R5A-2C-R2 compiled topology primitive audit")
    parser.add_argument("--pytest-result", type=Path, default=None)
    args = parser.parse_args(argv)
    final, closure_domain, shortcut_catalog, generator, shortcuts, evaluation, ablations = build_report(args.pytest_result)
    write_artifacts(final, closure_domain, shortcut_catalog, generator, shortcuts, evaluation, ablations)
    written = load_json(FINAL_REPORT_PATH)
    print(json.dumps({"status": written["status"], "report": str(FINAL_REPORT_PATH), "report_body_sha256_excluding_timestamp": written["report_body_sha256_excluding_timestamp"]}, sort_keys=True))
    return 0 if written["status"] != INCONCLUSIVE_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
