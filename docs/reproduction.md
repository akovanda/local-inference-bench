# Reproduction

## Rebuild saved evidence

Follow the README to extract and verify the evidence archive. Both report commands accept `--evidence-root` (the directory containing `results/`) and `--output-dir`. Use fresh output directories when comparing releases:

```bash
python -m runtime.model_evidence_report_20261023 \
  --evidence-root . --output-dir build/cross-study
python -m runtime.public_model_expansion_report \
  --evidence-root . --output-dir build/expansion
python tools/release_evidence.py check-reports --root . --rebuilt build
```

Generation timestamps and public export paths may differ from the historical HTML. The comparison checks the supported report invariants against the curated evidence rather than requiring byte-identical HTML. Report generation uses the Python standard library and does not run graders or generated model code.

## Run endpoint contracts

`python bench.py --help` documents the general client. Supply an existing server, backend, served model alias, context, suite, and append-only output path. Use a new result path for a new configuration. The `quick`, `quality`, `context`, and `hard` suites are local contracts; they are distinct from public benchmarks. Recorded native throughput is available only when the server returns supported native timings.

## Public benchmark inputs and grading

The saved public manifest preserves task IDs, selection policy, input hashes, and upstream revisions. Existing outputs can be inspected without downloading the full grader environment. For a new public run, use `python -m runtime.public_benchmark_pilot run --help`; use `grade --help` for its separate grading inputs and process requirements.

The historical preparation script expects a clean EvalPlus checkout in `vendor/public-benchmarks/evalplus-v0.3.1`. If reproducing that preparation, acquire the official repository explicitly and pin it:

```bash
mkdir -p vendor/public-benchmarks
git clone https://github.com/evalplus/evalplus.git vendor/public-benchmarks/evalplus-v0.3.1
git -C vendor/public-benchmarks/evalplus-v0.3.1 checkout --detach e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2
python -m runtime.prepare_public_benchmark_pilot \
  --output-dir results/new-public-manifest --count 32 --seed 20261023
```

This command also acquires pinned Google Research IFEval sources and HumanEval+ data. Preserve its source hashes, review upstream licenses, and install the grader's dependencies in a dedicated environment. EvalPlus executes generated Python, so grade in an isolated container with resource limits. The archived grader build metadata records the historical environment; root package installation intentionally does not install or execute graders.

| Upstream | Captured pin |
| --- | --- |
| EvalPlus | `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2` (`v0.3.1`) |
| HumanEval+ | `v0.1.10`; compressed SHA256 `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101` |
| Google Research IFEval | `e6890f85757dd84e27ca6df2dd30651dafad28e0`; input SHA256 `67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49` |
| Strata | `477ac13cbe8e582c20484871ce459d20d82df09d` from `Niko1221/Strata` |

## Hardware reproduction

The runtime scripts include NVIDIA T4 placement experiments, Particle Tachyon QCM6490 CPU/OpenCL probes, and QNN/ONNX Runtime investigations. They are lab tooling, with captured process, port, affinity, filesystem, and vendor-runtime assumptions. Read a script and its corresponding evidence before running it on another machine. Some scripts intentionally stop or replace local inference services as part of controlled trials.

No model weights, Qualcomm SDK libraries, device binaries, or full upstream engines are redistributed. Acquire models from traceable publishers and pin revisions and file hashes. The QNN community export is an experimental third-party conversion, not an official LiquidAI deployment package. Its source and base-model lineage are recorded in the investigation; the capability result does not approve other community artifacts by association.

The lightweight Strata expert-profile test fixture is retained with its MIT license. Full Strata and llama.cpp builds are optional external dependencies; their separate version, license, and build requirements remain in force. The local gateway similarly requires `python -m pip install -e '.[gateway]'` and an explicitly reviewed configuration.
