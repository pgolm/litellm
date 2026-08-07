"""
Tests for Black Forest Labs common_utils — specifically assert_bfl_polling_url.

BFL uses regional subdomains (e.g. gateway.bfl.ai) for polling URLs that
differ from the submission host (api.bfl.ai). These tests verify that the
domain-aware check accepts legitimate BFL subdomains while still rejecting
off-domain and non-HTTPS URLs.
"""

import json
from pathlib import Path

import pytest

from litellm.llms.black_forest_labs.common_utils import (
    FLUX_2_MODELS,
    IMAGE_EDIT_MODELS,
    IMAGE_GENERATION_MODELS,
    BlackForestLabsError,
    assert_bfl_polling_url,
    build_flux_2_request_body,
    flux_2_reference_image_field,
    get_flux_2_model_spec,
)

REPO_ROOT = Path(__file__).parents[4]


class TestAssertBflPollingUrl:
    # --- should pass ---

    def test_exact_registered_domain(self):
        assert_bfl_polling_url("https://bfl.ai/v1/get_result?id=abc")

    def test_api_subdomain(self):
        assert_bfl_polling_url("https://api.bfl.ai/v1/get_result?id=abc")

    def test_gateway_subdomain(self):
        # BFL uses gateway.bfl.ai for polling — this was the original bug trigger
        assert_bfl_polling_url("https://gateway.bfl.ai/v1/get_result?id=abc")

    def test_regional_subdomain(self):
        assert_bfl_polling_url("https://eu.api.bfl.ai/v1/get_result?id=abc")

    def test_deep_subdomain(self):
        assert_bfl_polling_url("https://region.gateway.bfl.ai/poll?id=xyz")

    # --- should raise BlackForestLabsError ---

    def test_rejects_http_scheme(self):
        # HTTP must be rejected — x-key would be forwarded in plaintext
        with pytest.raises(BlackForestLabsError, match="scheme must be https"):
            assert_bfl_polling_url("http://api.bfl.ai/v1/get_result?id=abc")

    def test_rejects_off_domain(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://evil.com/steal-key")

    def test_rejects_lookalike_domain(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://notbfl.ai/v1/get_result?id=abc")

    def test_rejects_bfl_ai_as_suffix_only(self):
        # "fakebfl.ai" must not match — the check is on registered domain boundary
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://fakebfl.ai/v1/get_result?id=abc")

    def test_rejects_bfl_in_path(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://evil.com/bfl.ai/steal")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(BlackForestLabsError, match="scheme must be https"):
            assert_bfl_polling_url("ftp://api.bfl.ai/v1/get_result?id=abc")

    def test_rejects_javascript_scheme(self):
        with pytest.raises(BlackForestLabsError, match="scheme must be https"):
            assert_bfl_polling_url("javascript://api.bfl.ai/alert(1)")


EXPECTED_FLUX_2_MODELS = {
    "flux-2-max",
    "flux-2-pro",
    "flux-2-pro-preview",
    "flux-2-flex",
    "flux-2-klein-9b",
    "flux-2-klein-9b-preview",
    "flux-2-klein-4b",
}


class TestFlux2Registry:
    def test_all_flux_2_endpoints_are_registered(self):
        assert set(FLUX_2_MODELS) == EXPECTED_FLUX_2_MODELS

    @pytest.mark.parametrize("model", sorted(EXPECTED_FLUX_2_MODELS))
    def test_registered_for_both_generation_and_edit(self, model):
        """One endpoint per variant serves text-to-image and editing alike."""
        endpoint = f"/v1/{model}"

        assert IMAGE_GENERATION_MODELS[model] == endpoint
        assert IMAGE_EDIT_MODELS[model] == endpoint

    @pytest.mark.parametrize("model", sorted(EXPECTED_FLUX_2_MODELS))
    def test_priced_in_the_model_cost_map(self, model):
        cost_map = json.loads((REPO_ROOT / "model_prices_and_context_window.json").read_text())
        entry = cost_map[f"black_forest_labs/{model}"]

        assert entry["litellm_provider"] == "black_forest_labs"
        assert entry["output_cost_per_image"] > 0
        assert set(entry["supported_endpoints"]) == {"/v1/images/generations", "/v1/images/edits"}

    def test_spec_lookup_strips_the_provider_prefix(self):
        assert get_flux_2_model_spec("black_forest_labs/flux-2-flex") is FLUX_2_MODELS["flux-2-flex"]
        assert get_flux_2_model_spec("FLUX-2-FLEX") is FLUX_2_MODELS["flux-2-flex"]

    def test_flux_1_models_have_no_flux_2_spec(self):
        assert get_flux_2_model_spec("flux-kontext-pro") is None
        assert get_flux_2_model_spec("flux-pro-1.1-ultra") is None

    def test_reference_image_field_names(self):
        assert flux_2_reference_image_field(0) == "input_image"
        assert flux_2_reference_image_field(1) == "input_image_2"
        assert flux_2_reference_image_field(7) == "input_image_8"

    def test_non_boolean_prompt_upsampling_is_not_coerced(self):
        """A stray string must not silently flip disable_pup the wrong way."""
        body = build_flux_2_request_body(
            spec=FLUX_2_MODELS["flux-2-pro"],
            prompt="x",
            optional_params={"prompt_upsampling": "false"},
        )

        assert "disable_pup" not in body
        assert "prompt_upsampling" not in body
