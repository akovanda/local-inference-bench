# Third-party notices

The root MIT license applies to original code and documentation by Andrew Kovanda. It does not relicense upstream code, model weights, benchmark prompts, or dataset-derived material in the evidence archive. This release includes selected benchmark inputs/results and one small runtime fixture, with source receipts retained where available.

| Material | Source and captured revision | License and release treatment |
| --- | --- | --- |
| Strata expert profile fixture | [Niko1221/Strata](https://github.com/Niko1221/Strata), `477ac13cbe8e582c20484871ce459d20d82df09d` | MIT, copyright 2026 Niko1221 and the Strata contributors. Only the selected fixture and license are included; the full engine is excluded. |
| EvalPlus grading tool and referenced outputs | [evalplus/evalplus](https://github.com/evalplus/evalplus/tree/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2), `v0.3.1` | Apache-2.0. Full checkout excluded. Its `evalplus/eval/` code additionally retains the underlying HumanEval MIT terms. |
| HumanEval+ selected prompts/test metadata | [HumanEval+ release](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10), `v0.1.10` | Apache-2.0, as identified by the [release repository](https://github.com/evalplus/humanevalplus_release) and [EvalPlus dataset card](https://huggingface.co/datasets/evalplus/humanevalplus/blob/main/README.md). Underlying HumanEval material retains the OpenAI MIT notice. |
| IFEval selected prompts and grading metadata | [Google Research](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval), `e6890f85757dd84e27ca6df2dd30651dafad28e0` | Apache-2.0; copyright The Google Research Authors. Full checkout excluded. |
| llama.cpp/ggml runtime references | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT, copyright The ggml authors. Runtime binaries and full source checkouts excluded; captured pins are in per-study receipts. |
| Community LFM2.5-350M QNN export references | [PrismPhi/radxa-dragon-q6a-qcs6490-lfm2.5-350m-qnn-npu](https://github.com/PrismPhi/radxa-dragon-q6a-qcs6490-lfm2.5-350m-qnn-npu) | Upstream repository Apache-2.0; referenced as experimental third-party work. Exported model/SDK binaries and vendor code are excluded. Original model terms remain separate. |

Retained license texts are in [docs/licenses](docs/licenses): [Strata MIT](docs/licenses/Strata-MIT.txt), [EvalPlus Apache-2.0 with HumanEval note](docs/licenses/EvalPlus-Apache-2.0.txt), [Google Research Apache-2.0](docs/licenses/Google-Research-Apache-2.0.txt), [OpenAI HumanEval MIT](docs/licenses/HumanEval-MIT.txt), and [llama.cpp MIT](docs/licenses/llama.cpp-MIT.txt). The fixture's adjacent `vendor/strata/LICENSE` is also retained.

Models mentioned in reports are independently published artifacts. This repository redistributes no weights and grants no rights to upstream models. Quantizations, conversions, and runtime exports are distinguished from original model releases in source receipts; benchmark results are not an endorsement of every publisher or conversion. Qualcomm QNN/QAIRT SDKs, ONNX Runtime wheels, and device runtime libraries must be obtained separately under their applicable terms.

Python package dependencies and GitHub Actions are acquired separately and retain their own licenses. Adding a third-party file requires retaining its attribution and applicable license; a citation alone is not a replacement for a required notice.
