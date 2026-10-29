#!/usr/bin/env python3
"""Run one isolated Qwen 3.6 native CLI expert-cache diagnostic.

This runner intentionally accepts no executable, model, tokenizer, or config
paths from the command line.  It uses the pinned 35B Strata artifacts and emits
one compact JSON result plus the native engine log.  It never starts a server.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from .signal_exec import unblocked_exec_command
    from .strata_trial import (
        TrialError as SharedPreflightError,
        assert_preflight_clear,
        defer_termination_signals,
        install_signal_handlers,
    )
else:
    from signal_exec import unblocked_exec_command
    from strata_trial import (
        TrialError as SharedPreflightError,
        assert_preflight_clear,
        defer_termination_signals,
        install_signal_handlers,
    )


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STRATA_ROOT = ROOT / "vendor" / "strata"
CONFIG_PATH = ROOT / "runtime" / "qwen36-greedy-mtp0-32k.json"
ENGINE_PATH = Path("/external/storage/moe-bench/strata-build/strata")
MODEL_PATH = Path(
    "/external/storage/moe-bench/models/qwen36-UD-Q4_K_XL/"
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
)
TOKENIZER_PATH = ROOT / "runtime" / "strata-data" / "packs" / "qwen36-ud-q4_k_xl" / "tokenizer"
MAX_RUNTIME_SECONDS = 420.0
TERM_TIMEOUT_SECONDS = 20.0
KILL_TIMEOUT_SECONDS = 10.0
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
ALLOWED_CASES = ("quick_generation", "hard_transaction_ledger")

TIMING_RE = re.compile(
    r"^strata-q36: prompt (?P<prompt_n>\d+) tokens in (?P<prompt_ms>[0-9.]+) ms "
    r"\((?P<prompt_tps>[0-9.]+) tok/s\), (?P<generated_n>\d+) generated in "
    r"(?P<generated_ms>[0-9.]+) ms \((?P<generated_tps>[0-9.]+) tok/s\)$"
)
HYBRID_RE = re.compile(
    r"^strata-q36: hybrid experts: (?P<gpu>\d+) of (?P<total>\d+) batch entries "
    r"on the GPU \((?P<percent>[0-9.]+)%\), (?P<cpu>\d+) on the CPU pool in "
    r"(?P<cpu_ms>[0-9.]+) ms, (?P<replacements>\d+) slot replacements$"
)
PLACEMENT_RE = re.compile(
    r"^strata-q36: hybrid experts: (?P<slots>\d+) of (?P<experts>\d+) per layer in VRAM "
    r"\((?P<vram_gib>[0-9.]+) GiB\), all (?P<host_gib>[0-9.]+) GiB in "
    r"(?P<pinning>pinned|pageable) host memory, the rest of a batch of <= (?P<rows>\d+) "
    r"rows on (?P<threads>\d+) CPU threads, bigger prompt chunks streamed "
    r"\((?P<staging_gib>[0-9.]+) GiB staging\); (?P<init_seconds>[0-9.]+) s$"
)
GEN_RE = re.compile(r"^gen:(?P<ids>(?: \d+)*)$")


class TrialError(RuntimeError):
    pass


@dataclass(frozen=True)
class PinnedConfig:
    source_path: Path
    source_sha256: str
    engine_path: Path
    model_path: Path
    tokenizer_path: Path
    cwd: Path
    model_name: str
    library_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class PromptTokens:
    case_id: str
    case_prompt_sha256: str
    rendered_prompt_sha256: str
    rendered_prompt_chars: int
    ids: tuple[int, ...]
    ids_sha256: str
    tokenizer: Any


@dataclass
class ProcessHandle:
    process: subprocess.Popen[Any]
    pgid: int
    log_handle: Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrialError(f"pinned config is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise TrialError("pinned config must be a JSON object")
    return value, raw


def load_pinned_config(
    source_path: Path = CONFIG_PATH,
    *,
    expected_engine: Path = ENGINE_PATH,
    expected_model: Path = MODEL_PATH,
    expected_tokenizer: Path = TOKENIZER_PATH,
    expected_cwd: Path = STRATA_ROOT,
) -> PinnedConfig:
    """Load the one approved descriptor and reject any executable argument drift."""
    source_path = source_path.resolve()
    data, raw = _read_json_object(source_path)
    engine = Path(str(data.get("exe", ""))).resolve()
    cwd = Path(str(data.get("cwd", ""))).resolve()
    tokenizer = Path(str(data.get("tokenizer", ""))).resolve()
    expected_args = [
        "--native",
        str(expected_model.resolve()),
        "--max-context",
        "32768",
        "--prefill",
        "auto",
        "--mtp",
        "0",
    ]
    if engine != expected_engine.resolve():
        raise TrialError("pinned config engine path differs from the approved Qwen 35B engine")
    if cwd != expected_cwd.resolve():
        raise TrialError("pinned config cwd differs from the approved Strata checkout")
    if tokenizer != expected_tokenizer.resolve():
        raise TrialError("pinned config tokenizer differs from the approved Qwen 35B tokenizer")
    if data.get("args") != expected_args:
        raise TrialError("pinned config native arguments differ from the approved Qwen 35B settings")
    model_name = data.get("model_name")
    if not isinstance(model_name, str) or not model_name or "\x00" in model_name:
        raise TrialError("pinned config has no valid model_name")
    libraries = data.get("lib_dirs", [])
    if not isinstance(libraries, list) or any(
        not isinstance(value, str) or not value or "\x00" in value for value in libraries
    ):
        raise TrialError("pinned config lib_dirs must be a string list")
    library_dirs = tuple(Path(value).resolve() for value in libraries)
    for label, path, executable in (
        ("engine", engine, True),
        ("model", expected_model.resolve(), False),
    ):
        if not path.is_file() or not os.access(path, os.X_OK if executable else os.R_OK):
            raise TrialError(f"approved {label} is missing or inaccessible: {path}")
    if not tokenizer.is_dir() or not all(
        (tokenizer / name).is_file()
        for name in ("chat_template.jinja", "vocab.json", "merges.txt", "token_type.json")
    ):
        raise TrialError(f"approved tokenizer files are incomplete: {tokenizer}")
    if not cwd.is_dir():
        raise TrialError(f"approved Strata cwd is missing: {cwd}")
    return PinnedConfig(
        source_path=source_path,
        source_sha256=sha256_bytes(raw),
        engine_path=engine,
        model_path=expected_model.resolve(),
        tokenizer_path=tokenizer,
        cwd=cwd,
        model_name=model_name,
        library_dirs=library_dirs,
    )


def _load_tokenizer_and_template(tokenizer_path: Path) -> tuple[Any, Any]:
    tools_path = str(STRATA_ROOT / "tools")
    root_path = str(STRATA_ROOT)
    for path in (tools_path, root_path):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import strata_tokenizer as tokenizer_module
        from serve.frontend import ChatTemplate
    except ImportError as exc:
        raise TrialError(
            "the pinned Strata tokenizer frontend is not importable; run with "
            "vendor/strata/.venv/bin/python"
        ) from exc

    try:
        vocab = json.loads((tokenizer_path / "vocab.json").read_text(encoding="utf-8"))
        token_values: list[str | None] = [None] * len(vocab)
        for token, index in vocab.items():
            token_values[index] = token
        if any(token is None for token in token_values):
            raise ValueError("token IDs are not contiguous")
        merges = (tokenizer_path / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tokenizer_path / "token_type.json").read_text(encoding="utf-8"))
        tokenizer = tokenizer_module.Tokenizer(token_values, merges, types)
        template = ChatTemplate(tokenizer_path / "chat_template.jinja")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TrialError("the pinned tokenizer data is invalid") from exc
    return tokenizer, template


def _case_prompt(case_id: str) -> str:
    # A direct ``python /abs/path/runtime/expert_cache_trial.py`` launch puts
    # runtime/, rather than the repository root, on sys.path.
    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    try:
        from bench_cases import get_cases
    except ImportError as exc:
        raise TrialError("benchmark cases are not importable") from exc
    suite = "quick" if case_id == "quick_generation" else "hard"
    matches = get_cases(suite, [case_id])
    if len(matches) != 1 or matches[0].case_id != case_id:
        raise TrialError(f"fixed benchmark case is unavailable: {case_id}")
    return matches[0].prompt


def tokenize_case(case_id: str, tokenizer_path: Path) -> PromptTokens:
    prompt = _case_prompt(case_id)
    tokenizer, template = _load_tokenizer_and_template(tokenizer_path)
    try:
        rendered = template.render(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, parse_special=True)
    except Exception as exc:  # the pinned template/tokenizer can raise their own public errors
        raise TrialError("failed to render and tokenize the fixed benchmark prompt") from exc
    if not ids or any(not isinstance(value, int) or value < 0 for value in ids):
        raise TrialError("the fixed benchmark prompt produced invalid token IDs")
    encoded_ids = ",".join(str(value) for value in ids).encode("ascii")
    return PromptTokens(
        case_id=case_id,
        case_prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
        rendered_prompt_sha256=sha256_bytes(rendered.encode("utf-8")),
        rendered_prompt_chars=len(rendered),
        ids=tuple(ids),
        ids_sha256=sha256_bytes(encoded_ids),
        tokenizer=tokenizer,
    )


def build_engine_command(
    config: PinnedConfig,
    prompt: PromptTokens,
    *,
    context: int,
    generation_tokens: int,
    expert_slots: int | None,
) -> list[str]:
    if context not in (8192, 32768):
        raise TrialError("context must be 8192 or 32768")
    if generation_tokens not in (128, 256):
        raise TrialError("generation tokens must be 128 or 256")
    if expert_slots is not None and not 1 <= expert_slots <= 256:
        raise TrialError("expert slots must be between 1 and 256")
    if len(prompt.ids) + generation_tokens > context:
        raise TrialError("prompt plus generation length exceeds the selected context")
    command = [
        str(config.engine_path),
        "--native",
        str(config.model_path),
        "--max-context",
        str(context),
        "--prefill",
        "auto",
        "--mtp",
        "0",
        "--tokens",
        ",".join(str(value) for value in prompt.ids),
        "--n",
        str(generation_tokens),
    ]
    if expert_slots is not None:
        command.extend(("--expert-slots", str(expert_slots)))
    return command


def redacted_invocation(command: Sequence[str], prompt: PromptTokens) -> list[str]:
    result = list(command)
    try:
        index = result.index("--tokens") + 1
    except ValueError as exc:
        raise TrialError("native invocation has no token argument") from exc
    result[index] = f"<{len(prompt.ids)} token IDs; sha256={prompt.ids_sha256}>"
    return result


def engine_environment(
    config: PinnedConfig, base_environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    existing = environment.get("LD_LIBRARY_PATH")
    approved = [str(path) for path in config.library_dirs if path.is_dir()]
    if approved:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(approved + ([existing] if existing else []))
    return environment


def engine_environment_metadata(environment: Mapping[str, str]) -> dict[str, str]:
    return (
        {"STRATA_SM75_PERF": environment["STRATA_SM75_PERF"]}
        if "STRATA_SM75_PERF" in environment
        else {}
    )


def start_trial(
    command: Sequence[str], environment: Mapping[str, str], log_path: Path, *, cwd: Path
) -> ProcessHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_handle = log_path.open("xb", buffering=0)
    except FileExistsError as exc:
        raise TrialError(f"trial log already exists: {log_path}") from exc
    try:
        process = subprocess.Popen(
            unblocked_exec_command(command),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            env=dict(environment),
            start_new_session=True,
            close_fds=True,
        )
    except BaseException:
        log_handle.close()
        raise
    return ProcessHandle(process=process, pgid=process.pid, log_handle=log_handle)


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_group_gone(handle: ProcessHandle, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle.process.poll()
        if not _group_exists(handle.pgid):
            return True
        time.sleep(0.1)
    return not _group_exists(handle.pgid)


def cleanup_owned_process(handle: ProcessHandle) -> dict[str, Any]:
    """Stop only the process group created by this runner."""
    result = {"pgid": handle.pgid, "term_sent": False, "kill_sent": False, "stopped": False}
    try:
        if _group_exists(handle.pgid):
            os.killpg(handle.pgid, signal.SIGTERM)
            result["term_sent"] = True
        if not _wait_group_gone(handle, TERM_TIMEOUT_SECONDS):
            os.killpg(handle.pgid, signal.SIGKILL)
            result["kill_sent"] = True
            result["stopped"] = _wait_group_gone(handle, KILL_TIMEOUT_SECONDS)
        else:
            result["stopped"] = True
        try:
            handle.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        if not result["stopped"]:
            raise TrialError("owned native process group survived SIGKILL")
        return result
    finally:
        handle.log_handle.close()


def parse_native_output(text: str) -> dict[str, Any]:
    timing: dict[str, Any] | None = None
    hybrid: dict[str, Any] | None = None
    placement: dict[str, Any] | None = None
    generated_ids: list[int] | None = None
    native_lines: list[str] = []
    for line in text.splitlines():
        match = TIMING_RE.fullmatch(line)
        if match:
            native_lines.append(line)
            timing = {
                "prompt_tokens": int(match.group("prompt_n")),
                "prompt_ms": float(match.group("prompt_ms")),
                "prompt_tokens_per_second": float(match.group("prompt_tps")),
                "generated_tokens": int(match.group("generated_n")),
                "generated_ms": float(match.group("generated_ms")),
                "generated_tokens_per_second": float(match.group("generated_tps")),
            }
            continue
        match = HYBRID_RE.fullmatch(line)
        if match:
            native_lines.append(line)
            hybrid = {
                "gpu_batch_entries": int(match.group("gpu")),
                "total_batch_entries": int(match.group("total")),
                "gpu_percent_reported": float(match.group("percent")),
                "cpu_pool_batch_entries": int(match.group("cpu")),
                "cpu_pool_accumulated_ms": float(match.group("cpu_ms")),
                "slot_replacements": int(match.group("replacements")),
                "scope_note": "Counts include prompt rows and decode steps.",
                "cpu_ms_note": (
                    "CPU pool milliseconds are accumulated pool latency and may overlap GPU work; "
                    "they are not a wall-time fraction."
                ),
            }
            continue
        match = PLACEMENT_RE.fullmatch(line)
        if match:
            native_lines.append(line)
            placement = {
                "slots_per_layer": int(match.group("slots")),
                "experts_per_layer": int(match.group("experts")),
                "slot_cache_vram_gib": float(match.group("vram_gib")),
                "all_experts_host_gib": float(match.group("host_gib")),
                "host_memory": match.group("pinning"),
                "cpu_pool_max_batch_rows": int(match.group("rows")),
                "cpu_threads": int(match.group("threads")),
                "prompt_staging_gib": float(match.group("staging_gib")),
                "placement_init_seconds": float(match.group("init_seconds")),
                "slots_note": "The slot count is per routed-expert layer.",
            }
            continue
        match = GEN_RE.fullmatch(line)
        if match:
            generated_ids = [int(value) for value in match.group("ids").split()]
    return {
        "native_timing": timing,
        "hybrid_expert_placement": placement,
        "hybrid_expert_counters": hybrid,
        "generated_token_ids": generated_ids,
        "native_stats_lines": native_lines,
    }


def decoded_final_text(tokenizer: Any, generated_ids: Sequence[int]) -> tuple[str | None, str | None]:
    """Decode the first final-message segment and refuse to persist reasoning markup."""
    try:
        text = tokenizer.decode(list(generated_ids))
    except Exception:
        return None, "tokenizer decode failed"
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    if "<think>" in text or "</think>" in text:
        match = re.fullmatch(r"\s*<think>\s*</think>\s*(.*)", text, flags=re.DOTALL)
        if not match or "<think>" in match.group(1) or "</think>" in match.group(1):
            return None, "decoded output contained unexpected reasoning markup"
        text = match.group(1)
    return text, None


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise TrialError(f"could not read native trial log: {path}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise TrialError(f"trial result already exists: {path}") from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def result_paths(tag: str, results: Path = RESULTS) -> tuple[Path, Path]:
    return results / f"{tag}-expert-cache.json", results / f"{tag}-expert-cache.log"


def assert_results_unused(tag: str, results: Path = RESULTS) -> tuple[Path, Path]:
    result_path, log_path = result_paths(tag, results)
    collisions = [path for path in (result_path, log_path) if os.path.lexists(path)]
    if collisions:
        raise TrialError("trial tag would overwrite an existing artifact: " + ", ".join(map(str, collisions)))
    return result_path, log_path


def execute_trial(
    args: argparse.Namespace,
    config: PinnedConfig,
    prompt: PromptTokens,
    result_path: Path,
    log_path: Path,
) -> tuple[dict[str, Any], int]:
    command = build_engine_command(
        config,
        prompt,
        context=args.context,
        generation_tokens=args.tokens,
        expert_slots=args.expert_slots,
    )
    environment = engine_environment(config)
    started_at = utc_now()
    started = time.monotonic()
    handle: ProcessHandle | None = None
    returncode: int | None = None
    timed_out = False
    interrupted = False
    cleanup: dict[str, Any] | None = None
    try:
        with defer_termination_signals():
            handle = start_trial(command, environment, log_path, cwd=config.cwd)
        returncode = handle.process.wait(timeout=MAX_RUNTIME_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if handle is not None:
            with defer_termination_signals():
                cleanup = cleanup_owned_process(handle)
            if returncode is None:
                returncode = handle.process.poll()

    log_text = _read_log(log_path)
    parsed = parse_native_output(log_text)
    generated_ids = parsed.pop("generated_token_ids")
    final_text: str | None = None
    omitted_reason: str | None = None
    generated_hash: str | None = None
    if generated_ids is not None:
        generated_hash = sha256_bytes(",".join(map(str, generated_ids)).encode("ascii"))
        final_text, omitted_reason = decoded_final_text(prompt.tokenizer, generated_ids)
    timing = parsed["native_timing"]
    integrity = {
        "reported_prompt_tokens_match": bool(
            timing is not None and timing["prompt_tokens"] == len(prompt.ids)
        ),
        "reported_generated_tokens_match": bool(
            timing is not None and timing["generated_tokens"] == args.tokens
        ),
        "emitted_generated_ids_match": bool(
            generated_ids is not None and len(generated_ids) == args.tokens
        ),
        "hybrid_counts_sum": bool(
            parsed["hybrid_expert_counters"] is not None
            and parsed["hybrid_expert_counters"]["gpu_batch_entries"]
            + parsed["hybrid_expert_counters"]["cpu_pool_batch_entries"]
            == parsed["hybrid_expert_counters"]["total_batch_entries"]
        ),
    }
    success = (
        returncode == 0
        and not timed_out
        and not interrupted
        and parsed["hybrid_expert_placement"] is not None
        and parsed["hybrid_expert_counters"] is not None
        and all(integrity.values())
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "tag": args.tag,
        "started_at": started_at,
        "finished_at": utc_now(),
        "status": "passed" if success else "interrupted" if interrupted else "timed_out" if timed_out else "failed",
        "case_id": args.case,
        "thinking": "none",
        "context_max_tokens": args.context,
        "generation_tokens_requested": args.tokens,
        "expert_slots_per_layer_requested": args.expert_slots,
        "wall_seconds": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "cleanup": cleanup,
        "config": {
            "source_path": str(config.source_path),
            "source_sha256": config.source_sha256,
            "engine_path": str(config.engine_path),
            "model_path": str(config.model_path),
            "tokenizer_path": str(config.tokenizer_path),
            "model_name": config.model_name,
            "native_settings": {"prefill": "auto", "mtp": 0},
        },
        "engine_environment": engine_environment_metadata(environment),
        "invocation": redacted_invocation(command, prompt),
        "prompt": {
            "case_prompt_sha256": prompt.case_prompt_sha256,
            "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
            "rendered_prompt_chars": prompt.rendered_prompt_chars,
            "prompt_token_count": len(prompt.ids),
            "token_ids_sha256": prompt.ids_sha256,
            "chat_template_enable_thinking": False,
        },
        "integrity": integrity,
        **parsed,
        "generated_token_count": len(generated_ids) if generated_ids is not None else None,
        "generated_token_ids_sha256": generated_hash,
        "decoded_final_text": final_text,
        "decoded_output_omitted_reason": omitted_reason,
    }
    atomic_write_json(result_path, record)
    if interrupted:
        return record, 130
    if timed_out:
        return record, 124
    return record, 0 if success else 1


def expert_slots_value(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= parsed <= 256:
        raise argparse.ArgumentTypeError("must be between 1 and 256")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=ALLOWED_CASES, default="quick_generation")
    parser.add_argument("--tokens", type=int, choices=(128, 256), default=256)
    parser.add_argument("--expert-slots", type=expert_slots_value)
    parser.add_argument("--context", type=int, choices=(8192, 32768), default=32768)
    parser.add_argument("--tag", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not TAG_RE.fullmatch(args.tag):
        parser.error("--tag must contain only letters, numbers, dot, underscore, and hyphen")
    try:
        result_path, log_path = assert_results_unused(args.tag)
        config = load_pinned_config()
        prompt = tokenize_case(args.case, config.tokenizer_path)
        assert_preflight_clear()
        install_signal_handlers()
        record, code = execute_trial(args, config, prompt, result_path, log_path)
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "result": str(result_path),
                    "log": str(log_path),
                    "hybrid_expert_counters": record["hybrid_expert_counters"],
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return code
    except (TrialError, SharedPreflightError, OSError, subprocess.SubprocessError) as exc:
        print(f"expert_cache_trial: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
