import argparse
import contextlib
import io
import json
import select
import signal
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from runtime import gpu_window, native_trial, strata_trial


class _FakeEvents:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


class _ExitedProcess:
    pid = 4242

    def poll(self):
        return 0


class _RunningProcess:
    pid = 4343

    def poll(self):
        return None

    def wait(self, timeout=None):
        return -signal.SIGKILL


class GpuWindowTests(unittest.TestCase):
    def test_explicit_long_window_is_bounded_before_any_live_action(self):
        with mock.patch.object(gpu_window, "run_window", return_value=0) as run, mock.patch.object(
            gpu_window, "validate_trial", return_value=[sys.executable, "native_trial.py"]
        ), mock.patch.object(gpu_window, "EventLog", return_value=_FakeEvents()), mock.patch.object(
            gpu_window.signal, "signal"
        ):
            self.assertEqual(gpu_window.main(["--max-seconds", "600", "--", "native_trial.py"]), 0)
            self.assertEqual(run.call_args.args[2], 600)
            run.reset_mock()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                gpu_window.main(["--max-seconds", "601", "--", "native_trial.py"])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_long_session_opt_in_is_model_session_only_and_capped_at_24_hours(self):
        model_command = [sys.executable, str(gpu_window.MODEL_SESSION_TRIAL)]
        native_command = [
            sys.executable,
            str(gpu_window.ROOT / "runtime" / "native_trial.py"),
        ]
        with mock.patch.object(gpu_window, "run_window", return_value=0) as run, mock.patch.object(
            gpu_window, "validate_trial", side_effect=[model_command, model_command, model_command, native_command]
        ), mock.patch.object(gpu_window, "EventLog", return_value=_FakeEvents()), mock.patch.object(
            gpu_window.signal, "signal"
        ):
            self.assertEqual(
                gpu_window.main(
                    ["--long-session", "--max-seconds", "86400", "--", "runtime/model_session.py"]
                ),
                0,
            )
            self.assertEqual(run.call_args.args[2], 86400)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                gpu_window.main(
                    ["--long-session", "--max-seconds", "86401", "--", "runtime/model_session.py"]
                )
            self.assertEqual(raised.exception.code, 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                gpu_window.main(["--max-seconds", "601", "--", "runtime/model_session.py"])
            self.assertEqual(raised.exception.code, 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                gpu_window.main(
                    ["--long-session", "--max-seconds", "600", "--", "runtime/native_trial.py"]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(run.call_count, 1)

    @unittest.skipUnless(hasattr(signal, "pthread_sigmask"), "POSIX signal masks required")
    def test_exec_shims_unblock_term_and_int_before_real_child_exec(self):
        child_code = (
            "import json,signal,sys;"
            "signal.signal(signal.SIGTERM,lambda *_:sys.exit(23));"
            "blocked=signal.pthread_sigmask(signal.SIG_BLOCK,[]);"
            "print(json.dumps(sorted(int(item) for item in blocked)),flush=True);"
            "signal.pause()"
        )
        watched = {signal.SIGINT, signal.SIGTERM}
        builders = (
            gpu_window.unblocked_exec_command,
            native_trial.unblocked_exec_command,
            strata_trial.unblocked_exec_command,
        )
        for builder in builders:
            with self.subTest(builder=builder.__module__):
                previous = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
                try:
                    process = subprocess.Popen(
                        builder([sys.executable, "-c", child_code]),
                        stdout=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous)
                try:
                    readable, _, _ = select.select([process.stdout], [], [], 3)
                    self.assertTrue(readable, "child did not report its signal mask")
                    blocked = json.loads(process.stdout.readline())
                    self.assertNotIn(int(signal.SIGINT), blocked)
                    self.assertNotIn(int(signal.SIGTERM), blocked)
                    process.terminate()
                    self.assertEqual(process.wait(timeout=3), 23)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)
                    if process.stdout is not None:
                        process.stdout.close()

    def test_time_bounds_reject_nonfinite_values(self):
        for value in ("nan", "inf", "-inf", "0"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                gpu_window.positive_seconds(value)

    def test_unload_request_is_bound_to_original_model(self):
        with mock.patch.object(gpu_window.time, "monotonic", return_value=10.0), mock.patch.object(
            gpu_window, "_api_request"
        ) as request:
            gpu_window.request_model_unload(25.0)
        request.assert_called_once_with(
            "POST",
            "/api/generate",
            15.0,
            {"model": "gpt-oss-20b:64k", "keep_alive": 0, "stream": False},
        )

    def test_original_guard_requires_running_unpaused_container(self):
        for state, message in [
            ({"Running": True, "Paused": True}, "already paused"),
            ({"Running": False, "Paused": False}, "not running"),
        ]:
            with self.subTest(state=state), mock.patch.object(
                gpu_window, "original_state", return_value=state
            ):
                with self.assertRaisesRegex(gpu_window.WindowError, message):
                    gpu_window.require_original_available()

    def test_trial_allowlist_builds_argument_list_for_current_python(self):
        command = gpu_window.validate_trial(
            ["runtime/native_trial.py", "--tag", "a value; still one argument"]
        )
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(gpu_window.ROOT / "runtime" / "native_trial.py"))
        self.assertEqual(command[2:], ["--tag", "a value; still one argument"])
        llama_command = gpu_window.validate_trial(["runtime/llama_trial.py", "--tag", "qwen122"])
        self.assertEqual(llama_command[0], sys.executable)
        self.assertEqual(llama_command[1], str(gpu_window.ROOT / "runtime" / "llama_trial.py"))
        cache_command = gpu_window.validate_trial(["runtime/expert_cache_trial.py", "--tag", "cache-check"])
        self.assertEqual(cache_command[0], sys.executable)
        self.assertEqual(cache_command[1], str(gpu_window.ROOT / "runtime" / "expert_cache_trial.py"))
        session_command = gpu_window.validate_trial(["runtime/model_session.py", "--tag", "bulk"])
        self.assertEqual(session_command[0], sys.executable)
        self.assertEqual(session_command[1], str(gpu_window.MODEL_SESSION_TRIAL))
        with self.assertRaisesRegex(gpu_window.WindowError, "trial must be"):
            gpu_window.validate_trial(["bench.py"])

    def test_guard_failure_never_pauses_or_restores(self):
        events = _FakeEvents()
        state = gpu_window.SignalState()
        with mock.patch.object(
            gpu_window, "require_original_available", side_effect=gpu_window.WindowError("guard")
        ), mock.patch.object(gpu_window, "pause_original") as pause, mock.patch.object(
            gpu_window, "restore_original"
        ) as restore:
            result = gpu_window.run_window([sys.executable, "trial.py"], 1, 1, events, state)
        self.assertEqual(result, 1)
        pause.assert_not_called()
        restore.assert_not_called()

    def test_loaded_model_race_refuses_pause(self):
        events = _FakeEvents()
        state = gpu_window.SignalState()
        with mock.patch.object(gpu_window, "require_original_available"), mock.patch.object(
            gpu_window, "wait_for_idle"
        ), mock.patch.object(
            gpu_window, "loaded_models", return_value=[{"name": gpu_window.ORIGINAL_MODEL}]
        ), mock.patch.object(gpu_window, "pause_original") as pause, mock.patch.object(
            gpu_window, "restore_original"
        ) as restore, mock.patch.object(
            gpu_window, "defer_termination_signals", side_effect=contextlib.nullcontext
        ):
            result = gpu_window.run_window([sys.executable, "trial.py"], 1, 1, events, state)
        self.assertEqual(result, 1)
        pause.assert_not_called()
        restore.assert_not_called()

    def test_post_pause_gpu_race_restores_without_launching_trial(self):
        events = _FakeEvents()
        state = gpu_window.SignalState()
        busy = [[], [{"pid": 9, "name": "racer", "used_mib": 1024}]]
        with mock.patch.object(gpu_window, "require_original_available"), mock.patch.object(
            gpu_window, "wait_for_idle"
        ), mock.patch.object(gpu_window, "loaded_models", return_value=[]), mock.patch.object(
            gpu_window, "busy_gpu_processes", side_effect=busy
        ), mock.patch.object(gpu_window, "pause_original") as pause, mock.patch.object(
            gpu_window, "restore_original"
        ) as restore, mock.patch.object(gpu_window.subprocess, "Popen") as popen, mock.patch.object(
            gpu_window, "defer_termination_signals", side_effect=contextlib.nullcontext
        ):
            result = gpu_window.run_window([sys.executable, "trial.py"], 1, 1, events, state)
        self.assertEqual(result, 1)
        pause.assert_called_once_with()
        restore.assert_called_once_with(events)
        popen.assert_not_called()

    def test_launch_failure_restores_only_owned_pause(self):
        events = _FakeEvents()
        state = gpu_window.SignalState()
        with mock.patch.object(gpu_window, "require_original_available"), mock.patch.object(
            gpu_window, "wait_for_idle"
        ), mock.patch.object(gpu_window, "loaded_models", return_value=[]), mock.patch.object(
            gpu_window, "busy_gpu_processes", return_value=[]
        ), mock.patch.object(gpu_window, "pause_original") as pause, mock.patch.object(
            gpu_window, "restore_original"
        ) as restore, mock.patch.object(
            gpu_window.subprocess, "Popen", side_effect=OSError("launch failed")
        ), mock.patch.object(
            gpu_window, "defer_termination_signals", side_effect=contextlib.nullcontext
        ):
            result = gpu_window.run_window([sys.executable, "trial.py"], 1, 1, events, state)
        self.assertEqual(result, 1)
        pause.assert_called_once_with()
        restore.assert_called_once_with(events)
        self.assertIsNone(state.child)

    def test_successful_launch_uses_new_session_without_shell(self):
        events = _FakeEvents()
        state = gpu_window.SignalState()
        process = _ExitedProcess()
        command = [sys.executable, "/safe/trial.py", "--tag", "x"]
        with mock.patch.object(gpu_window, "require_original_available"), mock.patch.object(
            gpu_window, "wait_for_idle"
        ), mock.patch.object(gpu_window, "loaded_models", return_value=[]), mock.patch.object(
            gpu_window, "busy_gpu_processes", return_value=[]
        ), mock.patch.object(gpu_window, "pause_original"), mock.patch.object(
            gpu_window, "restore_original"
        ) as restore, mock.patch.object(
            gpu_window.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(
            gpu_window, "defer_termination_signals", side_effect=contextlib.nullcontext
        ):
            result = gpu_window.run_window(command, 1, 1, events, state)
        self.assertEqual(result, 0)
        popen.assert_called_once_with(
            gpu_window.unblocked_exec_command(command),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        self.assertNotIn("shell", popen.call_args.kwargs)
        restore.assert_called_once_with(events)

    def test_trial_cleanup_failure_blocks_restore_for_orphan_safety(self):
        events = _FakeEvents()
        state = gpu_window.SignalState()
        process = _RunningProcess()
        with mock.patch.object(gpu_window, "require_original_available"), mock.patch.object(
            gpu_window, "wait_for_idle"
        ), mock.patch.object(gpu_window, "loaded_models", return_value=[]), mock.patch.object(
            gpu_window, "busy_gpu_processes", return_value=[]
        ), mock.patch.object(gpu_window, "pause_original"), mock.patch.object(
            gpu_window, "restore_original"
        ) as restore, mock.patch.object(
            gpu_window.subprocess, "Popen", return_value=process
        ), mock.patch.object(
            gpu_window,
            "monitor_trial",
            side_effect=gpu_window.WindowError("monitor failed"),
        ), mock.patch.object(
            gpu_window,
            "terminate_owned_trial",
            side_effect=gpu_window.WindowError("cleanup failed"),
        ), mock.patch.object(
            gpu_window, "defer_termination_signals", side_effect=contextlib.nullcontext
        ):
            result = gpu_window.run_window([sys.executable, "trial.py"], 1, 1, events, state)
        self.assertEqual(result, 1)
        restore.assert_not_called()
        self.assertTrue(any(event == "trial_cleanup_failed" for event, _ in events.rows))
        self.assertTrue(any(event == "restore_blocked_orphan_guard" for event, _ in events.rows))

    def test_lingering_gpu_process_blocks_restore_after_trial_exit(self):
        events = _FakeEvents()
        state = gpu_window.SignalState()
        process = _ExitedProcess()
        busy = [[], [], [{"pid": 88, "name": "orphan", "used_mib": 2048}]]
        with mock.patch.object(gpu_window, "require_original_available"), mock.patch.object(
            gpu_window, "wait_for_idle"
        ), mock.patch.object(gpu_window, "loaded_models", return_value=[]), mock.patch.object(
            gpu_window, "busy_gpu_processes", side_effect=busy
        ), mock.patch.object(gpu_window, "pause_original"), mock.patch.object(
            gpu_window, "restore_original"
        ) as restore, mock.patch.object(
            gpu_window.subprocess, "Popen", return_value=process
        ), mock.patch.object(
            gpu_window, "defer_termination_signals", side_effect=contextlib.nullcontext
        ):
            result = gpu_window.run_window([sys.executable, "trial.py"], 1, 1, events, state)
        self.assertEqual(result, 1)
        restore.assert_not_called()
        guard = next(fields for event, fields in events.rows if event == "restore_blocked_orphan_guard")
        self.assertIn("pid 88", guard["reason"])
        self.assertTrue(guard["original_left_paused"])

    def test_signal_handler_forwards_to_owned_process_group(self):
        process = _RunningProcess()
        state = gpu_window.SignalState(child=process)
        with mock.patch.object(gpu_window.os, "killpg") as killpg:
            state.handle(signal.SIGTERM, None)
        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertEqual(state.received, signal.SIGTERM)

    def test_timeout_forwards_term_then_kills_after_cleanup_grace(self):
        process = _RunningProcess()
        events = _FakeEvents()
        state = gpu_window.SignalState(child=process)
        with mock.patch.object(
            gpu_window.time, "monotonic", side_effect=[0.0, 241.0, 277.0]
        ), mock.patch.object(gpu_window.time, "sleep"), mock.patch.object(
            gpu_window.os, "killpg"
        ) as killpg:
            result = gpu_window.monitor_trial(process, 240, state, events)
        self.assertEqual(result, 124)
        self.assertTrue(state.forced_kill)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(process.pid, signal.SIGTERM), mock.call(process.pid, signal.SIGKILL)],
        )
        self.assertIn(("trial_timeout", {"max_seconds": 240, "child_pgid": process.pid}), events.rows)

    def test_unpause_retries_transient_docker_failure(self):
        events = _FakeEvents()
        success = SimpleNamespace(returncode=0)
        with mock.patch.object(
            gpu_window,
            "_docker",
            side_effect=[gpu_window.WindowError("temporary"), success],
        ) as docker, mock.patch.object(
            gpu_window,
            "original_state",
            side_effect=[{"Running": True, "Paused": True}, {"Running": True, "Paused": False}],
        ), mock.patch.object(gpu_window.time, "sleep"):
            gpu_window.restore_original(events, attempts=2)
        self.assertEqual(docker.call_count, 2)
        self.assertEqual(events.rows[-1], ("original_restored", {"attempt": 2}))


if __name__ == "__main__":
    unittest.main()
