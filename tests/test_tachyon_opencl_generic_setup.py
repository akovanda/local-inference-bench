import subprocess
import sys

from runtime import tachyon_opencl_generic_setup as setup


def test_generic_build_disables_both_adreno_kernel_paths():
    base = setup.load_base()
    flags = setup.cmake_flags(base)
    assert "-DGGML_OPENCL=ON" in flags
    assert "-DGGML_OPENCL_EMBED_KERNELS=ON" in flags
    assert "-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF" in flags
    assert "-DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF" in flags
    assert "-DGGML_OPENCL_USE_ADRENO_KERNELS=ON" not in flags
    assert "-DCMAKE_SYSTEM_PROCESSOR=aarch64" in flags
    assert f"-DLLAMA_BUILD_COMMIT={setup.SOURCE_COMMIT}" in flags


def test_deploy_target_is_separate_and_command_does_not_touch_private_icd():
    assert setup.DEFAULT_TARGET.endswith("/llama-cpp-opencl-generic")
    command = setup.remote_deploy_command(setup.DEFAULT_TARGET)
    assert setup.DEFAULT_TARGET in command
    assert setup.PRIVATE_ICD not in command
    assert "qualcomm.icd" not in command


def test_direct_cli_entrypoint_is_import_safe():
    completed = subprocess.run(
        [sys.executable, str(setup.ROOT / "runtime/tachyon_opencl_generic_setup.py"), "--help"],
        cwd=setup.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--no-deploy" in completed.stdout
