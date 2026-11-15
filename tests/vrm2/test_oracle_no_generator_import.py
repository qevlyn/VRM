from __future__ import annotations

import inspect
import subprocess
import sys

import vrm2.oracle.model as oracle_model


def test_model_source_has_no_generator_execution_dependency_or_semantic_branch():
    source = inspect.getsource(oracle_model)
    forbidden = (
        "vrm2.data.operations",
        "execute_operation",
        "execute_program",
        "TranslateModule",
        "RecolorModule",
        "ReflectModule",
        "LeftmostSelectorModule",
        "if operator",
        "if selector",
    )
    assert all(term not in source for term in forbidden)


def test_runtime_model_import_does_not_load_generator_execution_module():
    code = (
        "import sys; import vrm2.oracle.model; "
        "assert 'vrm2.data.operations' not in sys.modules; "
        "assert 'vrm2.data.microworld' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
