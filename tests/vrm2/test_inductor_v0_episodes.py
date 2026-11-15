from __future__ import annotations

from vrm2.data.types import make_grid
from vrm2.inductor.episodes import assemble_episode_records
from vrm2.inductor.train import loo_views
from vrm2.oracle.types import OracleExample

import torch


def _example(index: int) -> OracleExample:
    grid = [[0] * 8 for _ in range(8)]
    grid[index // 8][index % 8] = 1
    output = [[0] * 8 for _ in range(8)]
    output[index // 8][(index + 1) % 8] = 1
    return OracleExample(
        input_grid=make_grid(grid),
        typed_program_ids=((1, 1, 1, 4), (2, 2, 2, 3)),
        target_output_grid=make_grid(output),
        intermediate_target_grids=(make_grid(grid), make_grid(output)),
        episode_seed=index,
        split="train",
        composition_signature="diagnostic",
        diagnostic_bucket="test",
    )


def test_episode_assembly_is_deterministic_distinct_and_non_reusing():
    examples = tuple(_example(index) for index in range(9))
    first = assemble_episode_records(examples, "train", 811000, (2,))
    second = assemble_episode_records(examples, "train", 811000, (2,))
    supports, queries, manifest = first
    assert first == second
    assert len(supports) == len(queries) == 2
    assert manifest["examples_consumed"] == 8
    assert manifest["examples_discarded"] == 1
    assert len(set(manifest["consumed_example_fingerprints"])) == 8
    for support, query in zip(supports, queries, strict=True):
        assert support.episode_id == query.episode_id
        assert support.target_typed_program_ids == query.evaluator_typed_program_ids


def test_leave_one_out_views_never_contain_the_omitted_pair():
    inputs = torch.arange(3 * 8 * 8).reshape(1, 3, 8, 8)
    outputs = inputs + 1000
    input_views, output_views = loo_views(inputs, outputs)
    assert input_views.shape == (3, 2, 8, 8)
    for omitted in range(3):
        assert not input_views[omitted].eq(inputs[0, omitted]).flatten(1).all(dim=1).any()
        assert not output_views[omitted].eq(outputs[0, omitted]).flatten(1).all(dim=1).any()
