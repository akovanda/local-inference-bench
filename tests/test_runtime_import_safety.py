import importlib

import pytest


def test_download_qwen36_import_does_not_load_vendor_or_download(tmp_path):
    module = importlib.import_module("runtime.download_qwen36")
    module = importlib.reload(module)
    with pytest.raises(RuntimeError, match="optional vendor/strata checkout"):
        module.load_strata_setup(tmp_path)


def test_probe_import_does_not_parse_arguments_or_make_request():
    module = importlib.import_module("runtime.probe")
    module = importlib.reload(module)
    assert callable(module.main)
    assert module.build_parser().get_default("endpoint").startswith("http://127.0.0.1:")

