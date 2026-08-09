import json

import pytest

from pyodsp.alg import params


@pytest.fixture
def restore_params():
    original = {name: getattr(params, name) for name in _PARAM_NAMES}
    yield
    for name, value in original.items():
        setattr(params, name, value)


_PARAM_NAMES = [
    "BM_ABS_TOLERANCE",
    "BM_REL_TOLERANCE",
    "BM_TIME_LIMIT",
    "BM_SLACK_TOLERANCE",
    "BM_MAX_CUT_AGE",
    "BM_CUT_SIM_TOLERANCE",
    "BM_PURGE_FREQ",
    "BM_DUMMY_BOUND",
    "PBM_ML",
    "PBM_MR",
    "PBM_U_MIN",
    "PBM_E_S",
    "BM_LAMBDA_BOUND",
    "DEC_CUT_ABS_TOL",
    "SDDP_REL_TOLERANCE",
    "SDDP_IMPROVE_TOLERANCE",
]


def test_load_params_from_file_overrides_given_values(tmp_path, restore_params):
    file_path = tmp_path / "params.json"
    file_path.write_text(json.dumps({"BM_ABS_TOLERANCE": 0.5, "PBM_ML": 0.25}))

    params.load_params_from_file(str(file_path))

    assert params.BM_ABS_TOLERANCE == 0.5
    assert params.PBM_ML == 0.25


def test_load_params_from_file_keeps_defaults_for_missing_keys(tmp_path, restore_params):
    file_path = tmp_path / "params.json"
    file_path.write_text(json.dumps({"BM_ABS_TOLERANCE": 0.5}))
    default_rel_tolerance = params.BM_REL_TOLERANCE

    params.load_params_from_file(str(file_path))

    assert params.BM_REL_TOLERANCE == default_rel_tolerance


def test_load_params_from_file_missing_file_keeps_defaults(tmp_path, restore_params, capsys):
    missing_path = tmp_path / "does_not_exist.json"
    default_tolerance = params.BM_ABS_TOLERANCE

    params.load_params_from_file(str(missing_path))

    assert params.BM_ABS_TOLERANCE == default_tolerance
    assert "not found" in capsys.readouterr().out


def test_load_params_from_file_malformed_json_keeps_defaults(tmp_path, restore_params, capsys):
    file_path = tmp_path / "params.json"
    file_path.write_text("{not valid json")
    default_tolerance = params.BM_ABS_TOLERANCE

    params.load_params_from_file(str(file_path))

    assert params.BM_ABS_TOLERANCE == default_tolerance
    assert "Error decoding JSON" in capsys.readouterr().out
