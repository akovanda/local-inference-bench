import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "runtime" / "tachyon_runtime_setup.py"
SPEC = importlib.util.spec_from_file_location("tachyon_runtime_setup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def test_package_copies_only_requested_binaries_and_records_real_hashes(tmp_path):
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    expected = {}
    for index, name in enumerate(setup.BINARIES):
        payload = f"binary-{index}".encode()
        (build / "bin" / name).write_bytes(payload)
        expected[f"bin/{name}"] = hashlib.sha256(payload).hexdigest()
    (build / "bin" / "unrelated-tool").write_bytes(b"not packaged")

    bundle = tmp_path / "bundle"
    assert setup.package(build, bundle) == expected

    assert sorted(path.name for path in (bundle / "bin").iterdir()) == sorted(
        setup.BINARIES
    )
    metadata = json.loads((bundle / "BUILD.json").read_text())
    assert metadata["source_commit"] == setup.SOURCE_COMMIT
    assert metadata["sha256"] == expected
    assert (bundle / "SHA256SUMS").read_text().splitlines() == [
        f"{expected[name]}  {name}" for name in sorted(expected)
    ]


@pytest.mark.parametrize("jobs", [0, 5])
def test_main_rejects_build_parallelism_outside_device_safe_limit(jobs):
    with pytest.raises(SystemExit, match="between 1 and 4"):
        setup.main(["--jobs", str(jobs), "--no-deploy"])


def test_remote_deploy_command_replaces_target_and_preserves_previous(tmp_path):
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "old-marker").write_text("old")
    bundle = tmp_path / "bundle"
    (bundle / "bin").mkdir(parents=True)
    (bundle / "bin" / "tool").write_text("new")
    archive = tmp_path / "bundle.tar"
    subprocess.run(
        ["tar", "-C", str(tmp_path), "-cf", str(archive), "bundle"], check=True
    )

    with archive.open("rb") as stream:
        subprocess.run(
            ["bash", "-c", setup.remote_deploy_command(str(target))],
            stdin=stream,
            check=True,
        )

    assert (target / "bin" / "tool").read_text() == "new"
    backups = list(tmp_path.glob("runtime.previous.*"))
    assert len(backups) == 1
    assert (backups[0] / "old-marker").read_text() == "old"
    assert not Path(f"{target}.new").exists()
