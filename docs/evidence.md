# Evidence guide

The `v0.1.0` evidence asset is `local-inference-bench-evidence-v0.1.0.tar.gz`, accompanied by `SHA256SUMS`. Extract it at the source repository root. `evidence-manifest.json` identifies the selected relative paths, file hashes, and export policy; `python tools/release_evidence.py verify --root .` verifies the exported files.

| Start here | Purpose |
| --- | --- |
| `results/model-evidence-20261023-v2/report.html` | Complete 817-record cross-study snapshot |
| `results/model-evidence-20261023-v2/evidence.csv` | Detailed records with explicit study and grading fields |
| `results/model-evidence-20261023-v2/runtime-status.json` | Counts, limitations, and paired public-pilot summary |
| `results/model-evidence-20261023-v2/evidence-exclusions-index.json` | Investigations retained outside the stable 817 rows |
| `results/public-benchmark-pilot-20261023-v1/final-paired-comparison.json` | Per-task GPT-OSS/LFM latency and quality comparisons |
| `results/public-benchmark-pilot-20261023-v1/official-grader-caveats.json` | Unchanged IFEval grade and checker caveat |
| `results/public-model-expansion-20261024-v1/report/report.html` | Partial, interrupted expansion; read its status first |
| `results/tachyon-qnn-investigation-20261024-v1/assessment.json` | Later correction and scoped QNN capability assessment |
| `results/tachyon-qnn-investigation-20261024-v1/lfm350-pilot/latest-pair/htp-execution-proof.json` | Latest HTP execution proof and fallback checks |

HTML files are local reports. GitHub's source/raw viewer displays their source rather than the interactive page; download the evidence and open or locally serve the files as described in the README. Follow links relative to the extracted repository, and consult the manifest when a larger historical artifact was deliberately omitted.

The October 23 report is a historical snapshot. Its original calculator-only QNN conclusion is superseded in scope by the October 24 assessment. Keep both timestamps and claims visible when citing this work. The later evidence does not retroactively add new quality rows to the 817-record snapshot.

Archive hashes identify exported bytes. Source receipts inside historical records may contain hashes of pre-export files. Path sanitization and export metadata can change bytes, so those source hashes describe the captured originals; the release manifest describes the public export. Model identifiers and numerical observations remain attributable to their study rather than silently merged.
