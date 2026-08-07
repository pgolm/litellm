"""
Unit tests for Black Forest Labs image generation transformation functionality.

Note: Polling tests are now in test_bfl_image_generation_handler.py
since polling logic was moved to the handler.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.black_forest_labs.image_generation.transformation import (
    BlackForestLabsImageGenerationConfig,
    get_black_forest_labs_image_generation_config,
)
from litellm.llms.black_forest_labs.common_utils import BlackForestLabsError
from litellm.types.utils import ImageObject, ImageResponse


class TestBlackForestLabsImageGenerationTransformation:
    """
    Unit tests for Black Forest Labs image generation transformation functionality.
    """

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = BlackForestLabsImageGenerationConfig()
        self.model = "flux-pro-1.1"
        self.logging_obj = MagicMock()
        self.prompt = "A beautiful sunset over the ocean"

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are returned correctly."""
        params = self.config.get_supported_openai_params(self.model)

        assert "n" in params
        assert "size" in params
        assert "quality" in params

    def test_map_openai_params_basic(self):
        """Test mapping of OpenAI params to BFL params."""
        non_default_params = {}
        optional_params = {}

        result = self.config.map_openai_params(non_default_params, optional_params, self.model, drop_params=False)

        # Empty input should return empty output
        assert result == {}

    def test_map_openai_params_size_mapping(self):
        """Test that OpenAI size is mapped to BFL width/height."""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}

        result = self.config.map_openai_params(non_default_params, optional_params, self.model, drop_params=False)

        assert result["width"] == 1024
        assert result["height"] == 1024

    def test_map_openai_params_size_custom(self):
        """Test custom size parsing."""
        non_default_params = {"size": "800x600"}
        optional_params = {}

        result = self.config.map_openai_params(non_default_params, optional_params, self.model, drop_params=False)

        assert result["width"] == 800
        assert result["height"] == 600

    def test_map_openai_params_n_for_ultra(self):
        """Test that n is mapped to num_images for ultra model."""
        non_default_params = {"n": 4}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, "flux-pro-1.1-ultra", drop_params=False
        )

        assert result["num_images"] == 4

    def test_map_openai_params_quality_hd_for_ultra(self):
        """Test that 'hd' quality maps to raw=True for ultra model."""
        non_default_params = {"quality": "hd"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, "flux-pro-1.1-ultra", drop_params=False
        )

        assert result["raw"] is True

    def test_map_openai_params_unsupported_raises(self):
        """Test that unsupported params raise ValueError when drop_params=False."""
        non_default_params = {"unsupported_param": "value"}
        optional_params = {}

        with pytest.raises(ValueError, match="not supported"):
            self.config.map_openai_params(non_default_params, optional_params, self.model, drop_params=False)

    def test_map_openai_params_unsupported_dropped(self):
        """Test that unsupported params are dropped when drop_params=True."""
        non_default_params = {"unsupported_param": "value"}
        optional_params = {}

        result = self.config.map_openai_params(non_default_params, optional_params, self.model, drop_params=True)

        assert "unsupported_param" not in result

    def test_validate_environment_with_api_key(self):
        """Test that validate_environment sets headers correctly."""
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test_api_key",
        )

        assert result["x-key"] == "test_api_key"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """Test that validate_environment raises error when API key is missing."""
        headers = {}

        with patch(
            "litellm.llms.black_forest_labs.image_generation.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(BlackForestLabsError, match="BFL_API_KEY"):
                self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    messages=[],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                )

    def test_get_model_endpoint_flux_pro_1_1(self):
        """Test endpoint for flux-pro-1.1 model."""
        endpoint = self.config._get_model_endpoint("flux-pro-1.1")
        assert endpoint == "/v1/flux-pro-1.1"

    def test_get_model_endpoint_flux_pro_1_1_ultra(self):
        """Test endpoint for flux-pro-1.1-ultra model."""
        endpoint = self.config._get_model_endpoint("flux-pro-1.1-ultra")
        assert endpoint == "/v1/flux-pro-1.1-ultra"

    def test_get_model_endpoint_flux_dev(self):
        """Test endpoint for flux-dev model."""
        endpoint = self.config._get_model_endpoint("flux-dev")
        assert endpoint == "/v1/flux-dev"

    def test_get_model_endpoint_flux_pro(self):
        """Test endpoint for flux-pro model."""
        endpoint = self.config._get_model_endpoint("flux-pro")
        assert endpoint == "/v1/flux-pro"

    def test_get_model_endpoint_flux_kontext_pro(self):
        """Test endpoint for flux-kontext-pro model (supports both generation and editing)."""
        endpoint = self.config._get_model_endpoint("flux-kontext-pro")
        assert endpoint == "/v1/flux-kontext-pro"

    def test_get_model_endpoint_flux_kontext_max(self):
        """Test endpoint for flux-kontext-max model (supports both generation and editing)."""
        endpoint = self.config._get_model_endpoint("flux-kontext-max")
        assert endpoint == "/v1/flux-kontext-max"

    def test_get_model_endpoint_unknown_raises(self):
        """Test that unknown models raise ValueError."""
        with pytest.raises(ValueError, match="Unknown BFL image generation model"):
            self.config._get_model_endpoint("unknown-model")

    def test_get_model_endpoint_with_provider_prefix(self):
        """Test that provider prefix is stripped from model name."""
        endpoint = self.config._get_model_endpoint("black_forest_labs/flux-pro-1.1")
        assert endpoint == "/v1/flux-pro-1.1"

    def test_get_complete_url(self):
        """Test URL construction with default base."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="flux-pro-1.1",
            optional_params={},
            litellm_params={},
        )

        assert "https://api.bfl.ai/v1/flux-pro-1.1" == url

    def test_get_complete_url_custom_base(self):
        """Test URL construction with custom base."""
        url = self.config.get_complete_url(
            api_base="https://custom.api.com",
            api_key=None,
            model="flux-pro-1.1",
            optional_params={},
            litellm_params={},
        )

        assert "https://custom.api.com/v1/flux-pro-1.1" == url

    def test_transform_image_generation_request(self):
        """Test request body transformation."""
        request = self.config.transform_image_generation_request(
            model=self.model,
            prompt=self.prompt,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert request["prompt"] == self.prompt
        assert request["output_format"] == "png"

    def test_transform_image_generation_request_custom_format(self):
        """Test request body with custom output format."""
        request = self.config.transform_image_generation_request(
            model=self.model,
            prompt=self.prompt,
            optional_params={"output_format": "jpeg"},
            litellm_params={},
            headers={},
        )

        assert request["output_format"] == "jpeg"

    def test_transform_image_generation_request_ultra_params(self):
        """Test request body with ultra-specific params."""
        request = self.config.transform_image_generation_request(
            model="flux-pro-1.1-ultra",
            prompt=self.prompt,
            optional_params={
                "raw": True,
                "num_images": 2,
                "aspect_ratio": "16:9",
            },
            litellm_params={},
            headers={},
        )

        assert request["raw"] is True
        assert request["num_images"] == 2
        assert request["aspect_ratio"] == "16:9"

    def test_transform_image_generation_response_success(self):
        """Test response transformation with final polled response."""
        # The response is now the FINAL polled response from handler
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {"sample": "https://example.com/image.png"},
        }
        mock_response.status_code = 200

        model_response = ImageResponse(created=0, data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/image.png"

    def test_transform_image_generation_response_multiple_images(self):
        """Test response transformation with multiple images."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "Ready",
            "result": [
                "https://example.com/image1.png",
                "https://example.com/image2.png",
            ],
        }
        mock_response.status_code = 200

        model_response = ImageResponse(created=0, data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/image1.png"
        assert result.data[1].url == "https://example.com/image2.png"

    def test_transform_image_generation_response_no_image(self):
        """Test response transformation when no image URL is present."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {},
        }
        mock_response.status_code = 200

        model_response = ImageResponse(created=0, data=[])

        with pytest.raises(BlackForestLabsError, match="No image URL"):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_get_error_class(self):
        """Test that get_error_class returns BlackForestLabsError."""
        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={},
        )

        assert isinstance(error, BlackForestLabsError)
        assert error.status_code == 400
        assert "Test error" in str(error.message)

    def test_get_black_forest_labs_image_generation_config(self):
        """Test the factory function."""
        config = get_black_forest_labs_image_generation_config("flux-pro-1.1")

        assert isinstance(config, BlackForestLabsImageGenerationConfig)


FLUX_2_ENDPOINTS = {
    "flux-2-max": "/v1/flux-2-max",
    "flux-2-pro": "/v1/flux-2-pro",
    "flux-2-pro-preview": "/v1/flux-2-pro-preview",
    "flux-2-flex": "/v1/flux-2-flex",
    "flux-2-klein-9b": "/v1/flux-2-klein-9b",
    "flux-2-klein-9b-preview": "/v1/flux-2-klein-9b-preview",
    "flux-2-klein-4b": "/v1/flux-2-klein-4b",
}


class TestFlux2ImageGeneration:
    """
    FLUX.2 serves text-to-image and editing from one endpoint per variant, and each
    variant exposes a different parameter surface than FLUX.1. These tests pin the
    endpoints and the per-variant request bodies.
    """

    def setup_method(self):
        self.config = BlackForestLabsImageGenerationConfig()
        self.prompt = "A neon-lit alley in the rain"

    @pytest.mark.parametrize("model,endpoint", sorted(FLUX_2_ENDPOINTS.items()))
    def test_endpoint_per_model(self, model, endpoint):
        assert self.config._get_model_endpoint(model) == endpoint
        assert (
            self.config.get_complete_url(
                api_base=None,
                api_key=None,
                model=f"black_forest_labs/{model}",
                optional_params={},
                litellm_params={},
            )
            == f"https://api.bfl.ai{endpoint}"
        )

    def test_flex_exposes_step_control(self):
        """steps/guidance are the whole point of [flex] and must be settable."""
        params = self.config.get_supported_openai_params("flux-2-flex")

        assert "steps" in params
        assert "guidance" in params
        assert "prompt_upsampling" in params

    @pytest.mark.parametrize(
        "model", ["flux-2-max", "flux-2-pro", "flux-2-pro-preview", "flux-2-klein-9b", "flux-2-klein-4b"]
    )
    def test_step_control_is_flex_only(self, model):
        params = self.config.get_supported_openai_params(model)

        assert "steps" not in params
        assert "guidance" not in params

    @pytest.mark.parametrize("model", ["flux-2-klein-9b", "flux-2-klein-9b-preview", "flux-2-klein-4b"])
    def test_klein_has_no_prompt_upsampling_knob(self, model):
        """[klein] endpoints reject both prompt_upsampling and disable_pup."""
        assert "prompt_upsampling" not in self.config.get_supported_openai_params(model)

    @pytest.mark.parametrize("model", sorted(FLUX_2_ENDPOINTS))
    def test_flux_1_only_params_are_rejected(self, model):
        """aspect_ratio/raw/num_images/image_prompt_strength do not exist on FLUX.2."""
        params = self.config.get_supported_openai_params(model)

        assert "aspect_ratio" not in params
        assert "raw" not in params
        assert "num_images" not in params
        assert "image_prompt_strength" not in params

    def test_size_maps_to_width_and_height(self):
        optional_params = self.config.map_openai_params({"size": "1440x2048"}, {}, "flux-2-pro", drop_params=False)

        assert optional_params == {"width": 1440, "height": 2048}

    def test_request_body_for_pro_inverts_prompt_upsampling(self):
        """[pro]/[max] spell the upsampling knob as the inverted disable_pup."""
        body = self.config.transform_image_generation_request(
            model="black_forest_labs/flux-2-pro",
            prompt=self.prompt,
            optional_params={"width": 1440, "height": 2048, "prompt_upsampling": False, "seed": 7},
            litellm_params={},
            headers={},
        )

        assert body == {
            "prompt": self.prompt,
            "output_format": "png",
            "width": 1440,
            "height": 2048,
            "seed": 7,
            "disable_pup": True,
        }

    def test_request_body_for_pro_keeps_upsampling_enabled(self):
        body = self.config.transform_image_generation_request(
            model="flux-2-max",
            prompt=self.prompt,
            optional_params={"prompt_upsampling": True},
            litellm_params={},
            headers={},
        )

        assert body["disable_pup"] is False
        assert "prompt_upsampling" not in body

    def test_request_body_for_flex_passes_step_control(self):
        body = self.config.transform_image_generation_request(
            model="flux-2-flex",
            prompt=self.prompt,
            optional_params={"steps": 32, "guidance": 4.5, "prompt_upsampling": True},
            litellm_params={},
            headers={},
        )

        assert body["steps"] == 32
        assert body["guidance"] == 4.5
        assert body["prompt_upsampling"] is True
        assert "disable_pup" not in body

    def test_request_body_for_klein_drops_unsupported_fields(self):
        """Anything [klein] does not accept must not reach the wire, or BFL 422s."""
        body = self.config.transform_image_generation_request(
            model="flux-2-klein-4b",
            prompt=self.prompt,
            optional_params={"steps": 4, "guidance": 3.0, "prompt_upsampling": True, "seed": 1},
            litellm_params={},
            headers={},
        )

        assert body == {"prompt": self.prompt, "output_format": "png", "seed": 1}

    def test_request_body_drops_flux_1_only_fields(self):
        body = self.config.transform_image_generation_request(
            model="flux-2-pro",
            prompt=self.prompt,
            optional_params={"aspect_ratio": "16:9", "raw": True, "num_images": 4},
            litellm_params={},
            headers={},
        )

        assert body == {"prompt": self.prompt, "output_format": "png"}

    def test_output_format_can_be_overridden(self):
        body = self.config.transform_image_generation_request(
            model="flux-2-flex",
            prompt=self.prompt,
            optional_params={"output_format": "webp"},
            litellm_params={},
            headers={},
        )

        assert body["output_format"] == "webp"

    def test_flux_1_request_body_is_unchanged(self):
        """Regression guard: FLUX.1 keeps its own (permissive) parameter surface."""
        body = self.config.transform_image_generation_request(
            model="flux-pro-1.1-ultra",
            prompt=self.prompt,
            optional_params={"aspect_ratio": "16:9", "num_images": 2, "raw": True},
            litellm_params={},
            headers={},
        )

        assert body["aspect_ratio"] == "16:9"
        assert body["num_images"] == 2
        assert body["raw"] is True
        assert "disable_pup" not in body
