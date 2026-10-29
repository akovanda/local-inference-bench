"""Distinguish portable tests from checks requiring original lab artifacts.

Public derivatives intentionally have different hashes. Historical controllers
retain their original fail-closed pins; we do not rewrite them to bless edited
receipts. Use -m historical only in the original lab with original artifacts.
"""
import pytest


HISTORICAL_MODULES = {
    'test_bonsai_dspark_continuation.py',
    'test_flash_modes_sampling_parent_stage.py',
    'test_oss20b_public_retry_probe.py',
}
HISTORICAL_TESTS = {
    'test_frozen_pins_and_exact_only_poll0_config_difference',
    'test_real_strata_child_receipt_is_accepted_and_joins_persisted_telemetry',
    'test_generic_build_disables_both_adreno_kernel_paths',
    'test_captured_device_evidence_validates',
    'test_captured_qnn_250_device_evidence_validates_separately',
    'test_actual_latest_pair_handoff_is_exact_and_requires_discovery_opt_in',
    'test_actual_latest_result_proves_htp_and_rejects_qnn_cpu_mapping',
    'test_waiting_interrupt_resumes_by_archiving_unbound_failure',
    'test_published_parent_handoff_identifies_lfm26_not_first_plan_model',
}
UPSTREAM_TESTS = {
    'test_stage_retains_sha_proof_when_compatibility_inspection_fails',
    'test_actual_vendor_audit_pins_instruct_source_and_exact_assets',
    'test_asset_manifest_rejects_traversal_even_with_valid_hash_and_count',
    'test_direct_cli_audit_entrypoint',
}


def pytest_collection_modifyitems(items):
    for item in items:
        name = item.originalname or item.name
        if item.path.name in HISTORICAL_MODULES or name in HISTORICAL_TESTS:
            item.add_marker(pytest.mark.historical)
        if name in UPSTREAM_TESTS:
            item.add_marker(pytest.mark.optional_upstream)
