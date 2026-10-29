# Methods and interpretation

The unit of evidence is a saved request or diagnostic record with its study family, model/profile, suite, task ID, hardware, cap settings, grading definition, and source paths. Comparisons should keep those fields aligned. A model alias alone is insufficient to identify a configuration.

## Frozen public pilot

Before inference, each suite's official task IDs were ordered by `SHA256(seed:suite:official_task_id)`, using seed `20261023`; the first 32 were selected from 164 HumanEval+ and 541 IFEval tasks. Each model produced one sample per selected task. HumanEval prompts requested a complete function and code only; IFEval prompts were sent verbatim as user messages. This is a subset experiment, not a published full-benchmark score.

The T4 runs used 73,728-token context and 65,536-token output caps, a 21,600-second per-request timeout, temperature 0, seed `20261023`, and captured `medium` reasoning requests. GPT-OSS used the `gpt-oss-20b-mxfp4` alias; LFM used `lfm2.5-24b-a2b-qad-q4_0`. Templates and model/runtime implementations can interpret request controls differently, so identical request fields do not imply identical internal reasoning behavior.

HumanEval+ quality is EvalPlus's enhanced per-task test pass. IFEval quality is the official strict **prompt-level** pass. The sole LFM IFEval miss involved the checker's asterisk-delimited emphasis convention, despite the answer containing three Markdown headings. The official failure is preserved; no subjective replacement grade was inserted. IFEval tests instruction compliance, not factual knowledge or summary faithfulness.

## Study boundaries

| Family | Records | Interpretation |
| --- | ---: | --- |
| Legacy complete dashboard | 480 | Original mixed local suites and profiles; original semantics retained |
| T4 public 64-case pilot | 128 | Two model profiles × 64 tasks |
| Tachyon public screen | 48 | Three profiles × 16 tasks |
| Tachyon worker | 120 | Five profiles × 24 closed contracts |
| Tachyon latency | 41 | 24 A/B, 16 greedy diagnostics, 1 ordinary-default probe; repeated tasks |
| **Complete snapshot** | **817** | Distinct saved records, not independent cases |

Forty manual summary reviews and six accelerator evidence rows remain separate from the 817 records. The worker automatic summary check validates closed facts, keys, and word limits; it cannot replace a review of free-prose grounding. No aggregate quality score is pooled across families.

Tachyon CPU experiments used the QCM6490's CPU cores 4–7 for the CPU4 profile. Public screening used context/output caps of 16,384/8,192; worker and latency studies used 16,384/512. Tachyon public LFM screening used native thinking, while worker/latency variants used fast zero-reasoning-budget controls. Those profiles must not be conflated.

## Timing and incomplete observations

Native decode tokens/second measures server generation throughput. Wall time includes the path to a finished response and depends on output length, reasoning, prefill, transport, and cache state. The cross-study report weights native rates by server-reported `predicted_ms`; unavailable timings stay null. Failed accelerator probes have no throughput measurement, not a measured zero.

Use task-paired wall time for paired latency claims. GPT-OSS was faster for all 32 paired coding tasks; LFM was faster for 19/32 instruction tasks in the original T4 run. A larger aggregate decode rate does not establish faster useful completion.

Output-cap hits, truncated streams, request errors, missing grades, and pending cases remain visible. The expansion snapshot is interrupted, with 168/576 terminal cases, 407 pending, one error, and 2/9 profiles terminal. Its denominators and retry costs must be read from its own report; pending cases are not passes or failures.

## Runtime capability versus task quality

An accelerator test progresses through different claims: loading libraries, executing a calculator, executing a neural graph, generating text, and meeting task requirements. Each needs its own evidence. Earlier OpenCL failures did not prove layer offload. Later QNN runs did prove LFM2.5-350M generation on v68 with QNN 2.47 and 2.51; the latest run recorded HTP/FastRPC activity without CPU fallback. The three task answers were weak. That capability proof does not establish competitive quality, general NPU compatibility, or any LFM2.6 NPU result.

## Limits

This is a personal engineering investigation on a small hardware sample. Public subsets are small, one sample is used per task, several local contracts are bespoke, some latency records repeat tasks, and model quantization/profile choices differ. The archive is curated and sanitized rather than an image of the whole lab. Rebuilding reports reproduces the evidence transformation; reproducing inference also requires the captured hardware, software, models, and configuration. No new inferential claim is made about statistical significance or full-benchmark ranking.
