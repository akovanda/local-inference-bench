First public source and curated evidence release for Local Inference Bench, a personal investigation of useful local model responses on NVIDIA T4 and Particle Tachyon hardware.

This release includes endpoint benchmark clients, durable public-benchmark attempts, study orchestration and diagnostic tooling, portable report builders, and fixture-based CI. The separate evidence archive preserves selected observations, official grading outputs, source receipts, and historical HTML reports with public export paths and checksums.

The complete October 23 cross-study snapshot has 817 records across distinct tasks, profiles, repeats, and diagnostics. It is not a unified leaderboard. The October 24 model expansion is explicitly interrupted: 168/576 terminal cases, 407 pending, one error, and 2/9 profiles terminal. The later QNN investigation proves LFM2.5-350M v68 generation under QNN 2.47 and 2.51; its three weak task answers do not establish broad quality, and LFM2.6 NPU inference remains untested.

Download `local-inference-bench-evidence-v0.1.0.tar.gz` and `SHA256SUMS`, verify the checksum, and extract at the source repository root. Follow the README to verify the evidence manifest and rebuild reports. GitHub's raw HTML view does not render the reports; open downloaded HTML locally or serve it on loopback.

Original code and documentation are MIT licensed, copyright Andrew Kovanda 2026. Third-party prompts, datasets, and fixtures retain their original licenses; see `THIRD_PARTY_NOTICES.md`. Model weights, Qualcomm SDKs, large vendor trees, and private lab configuration are excluded.

Release checks: `tools/publish_release.sh --check` verifies and audits evidence, runs the portable fixture suite, rebuilds and compares both reports, and verifies the generated archive checksum. These checks reproduce report transformations of saved observations, not new inference runs.
