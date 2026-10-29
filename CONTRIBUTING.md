# Contributing

Small, focused improvements to runners, evidence validation, report correctness, and documentation are welcome. Open an issue with the concrete behavior, relevant script or artifact, expected result, and a minimal reproduction. For a result correction, include the study, model/profile, suite, task ID, grading definition, and source file.

Use Python 3.11+ and install `python -m pip install -e '.[test]'`. Run the portable selection in [validation](docs/validation.md), plus tests relevant to your change. Reports should retain pending, failed, capped, ungraded, and retry records and keep incompatible studies separate. Use synthetic fixture trees for new report tests where possible.

Do not add model weights, SDKs, credentials, full vendor repositories, or private endpoint configuration. Preserve upstream notices and record the source, revision, and hash for new external inputs. Use the evidence manifest and checksums for any proposed archive update; explain whether an observation, an export transformation, or a report interpretation changed.

Contributions to original code are provided under the repository's MIT license. Existing third-party material retains its own license. Submit a focused pull request describing the problem, resulting behavior, and validation. Hardware results should identify the hardware, runtime/model pins, sampling/template controls, context/output caps, and exact success criterion.
