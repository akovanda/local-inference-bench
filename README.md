# Local Inference Bench

An auditable record of local inference experiments on an NVIDIA T4 and a Particle Tachyon (Qualcomm QCM6490): durable runners, explicit task contracts, report builders, and curated evidence.

I built this project to answer a practical question: **which model and runtime produce a useful, correct response on the hardware I actually have?** Decode speed alone does not answer it. Reasoning length, output caps, prompt templates, grading rules, cache state, and device placement all change the outcome.

The project showcases my work in experiment design, resumable execution, runtime integration, failure diagnosis, and evidence reporting. I used AI coding assistance. The inference engines, model weights, and public benchmarks are upstream work; their authors retain credit and licenses. See [methods](docs/methodology.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## What the evidence says

The complete October 23 snapshot contains **817 records across distinct studies**, including repeats and diagnostics. It is not 817 independent tasks or one combined leaderboard.

| Evidence | Scope | What can be concluded |
| --- | --- | --- |
| T4 public pilot | Same frozen 32 HumanEval+ and 32 IFEval prompts per model | Task-specific quality and latency comparison under the captured profiles |
| Tachyon CPU screening | 8 HumanEval+ and 8 IFEval prompts per model | Small device-specific screen, not full-benchmark scores |
| Worker and latency studies | Closed routing, decision, and summary contracts; repeated cache/sampling probes | Performance on those contracts; free-prose summary review is separate |
| October 24 expansion | **Interrupted: 168/576 terminal cases, 407 pending, 1 error; 2/9 profiles terminal** | Partial observations only |
| Tachyon QNN investigation | Actual LFM2.5-350M generation on v68 under QNN 2.47 and 2.51 | Runtime capability proved; three weak task answers do not establish broad model quality |

In the frozen T4 pilot, GPT-OSS-20B scored **30/32 HumanEval+ and 26/32 IFEval**; LFM2.5-24B-A2B scored **31/32 and 31/32**. GPT-OSS finished faster on all 32 paired coding tasks, while LFM finished faster on 19/32 instruction tasks. Three original GPT-OSS instruction outputs hit the output cap. These are observations about the saved quantizations, templates, and reasoning settings, not universal model rankings. [Evidence guide](docs/evidence.md) identifies the source artifacts and caveats.

The October 24 QNN evidence corrects an earlier report's overly broad “no ready path” assessment: the later LFM2.5-350M run includes HTP/FastRPC execution proof without CPU fallback. **LFM2.6 NPU inference remains untested.**

## Quick start: inspect and rebuild the reports

Python 3.11+ is sufficient for report generation; no GPU, model download, or inference server is needed.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python bench.py --list-cases
```

Download `local-inference-bench-evidence-v0.1.0.tar.gz` and `SHA256SUMS` from the `v0.1.0` GitHub release into the repository root. The archive is distributed separately to keep source history small. Verify and extract it there:

```bash
sha256sum -c SHA256SUMS
tar -xzf local-inference-bench-evidence-v0.1.0.tar.gz
python tools/release_evidence.py verify --root .
python -m runtime.model_evidence_report_20261023 \
  --evidence-root . --output-dir build/cross-study
python -m runtime.public_model_expansion_report \
  --evidence-root . --output-dir build/expansion
python tools/release_evidence.py check-reports --root . --rebuilt build
```

Open `build/cross-study/report.html` or `build/expansion/report.html` locally. GitHub's raw HTML view does not render an interactive report. If your browser needs HTTP for linked artifacts, serve the repository locally:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8000/build/cross-study/report.html`. Keep the evidence at `results/…` relative to the repository root; the archive already uses that layout. Original HTML snapshots are also included in the archive.

## Run a new experiment

The general client talks to an existing Ollama or OpenAI-compatible server. For example, with your server already listening on port 8080:

```bash
python bench.py --backend openai --endpoint http://127.0.0.1:8080/v1 \
  --model YOUR_SERVED_MODEL --context 16384 --suite quick \
  --thinking none --output results/my-run.jsonl --skip-telemetry
```

This runs the project's small local contract suite, not HumanEval+ or IFEval. The public benchmark runner uses frozen manifests and separate official grading. Hardware setup scripts retain the assumptions of the original lab; review their paths, device access, and resource controls before running them. See [reproduction](docs/reproduction.md) for the boundaries between report rebuilding, endpoint benchmarking, and hardware reproduction.

## Code map

| Path | Responsibility |
| --- | --- |
| `bench.py`, `bench_cases.py` | Endpoint client, request/timing capture, local contract checks |
| `runtime/public_benchmark_pilot.py` | Durable per-task attempts and pinned official-grader integration |
| `runtime/public_model_expansion.py` | Profile orchestration and recovery for a larger public pilot |
| `runtime/*report*.py` | Evidence joins, explicit denominators, CSV and HTML reports |
| `runtime/tachyon_*` | CPU, OpenCL, QNN, and device-specific investigations |
| `local_general/` | Optional local model gateway; install with `.[gateway]` |
| `tools/release_evidence.py` | Curated evidence verification, audit, archive, and report comparison |
| `tests/` | Fixture-based unit tests plus evidence/hardware integration checks |

The repository excludes model weights, SDKs, large vendor checkouts, virtual environments, private operational configuration, and most raw workspace output. The release evidence manifest records the selected files and checksums. Sanitization changes identifying paths and operational details; it does not turn a partial study into a completed result.

## Validation and license

[CI](.github/workflows/ci.yml) runs a documented source-only test selection using synthetic fixtures. Full historical tests also cover lab files, frozen evidence, optional runtimes, and live services; they are not all portable to a bare clone. See [validation](docs/validation.md) for exact commands and scope.

Original code and documentation are [MIT licensed](LICENSE), copyright Andrew Kovanda, 2026. Benchmark prompts, upstream fixtures, and other third-party material retain their [original licenses](THIRD_PARTY_NOTICES.md). Cite the release using [CITATION.cff](CITATION.cff), together with the relevant benchmark and model sources.
