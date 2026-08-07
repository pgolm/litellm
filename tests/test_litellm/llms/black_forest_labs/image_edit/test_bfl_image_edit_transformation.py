"""
Unit tests for Black Forest Labs image edit transformation functionality.

Note: Polling tests are now in test_bfl_image_edit_handler.py
since polling logic was moved to the handler.
"""

import base64
import json
import os
import sys
import time
from io import BytesIO
from typing import Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.black_forest_labs.image_edit.transformation import (
    BlackForestLabsImageEditConfig,
)
from litellm.llms.black_forest_labs.common_utils import BlackForestLabsError
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageObject, ImageResponse


class TestBlackForestLabsImageEditTransformation:
    """
    Unit tests for Black Forest Labs image edit transformation functionality.
    """

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = BlackForestLabsImageEditConfig()
        self.model = "flux-kontext-pro"
        self.logging_obj = MagicMock()
        self.prompt = "Add a red hat to the person in the image"

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are returned correctly."""
        params = self.config.get_supported_openai_params(self.model)

        # BFL image edit supports BFL-specific params passed through directly
        assert isinstance(params, list)
        assert len(params) > 0
        assert "seed" in params
        assert "output_format" in params
        assert "safety_tolerance" in params

    def test_map_openai_params_basic(self):
        """Test mapping of OpenAI params to BFL params."""
        optional_params = ImageEditOptionalRequestParams()

        result = self.config.map_openai_params(
            image_edit_optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should have default output_format
        assert result.get("output_format") == "png"

    def test_map_openai_params_with_bfl_specific(self):
        """Test that BFL-specific params are passed through."""
        # BFL-specific params are passed as dict keys
        optional_params: ImageEditOptionalRequestParams = {
            "seed": 42,
            "safety_tolerance": 2,
            "aspect_ratio": "16:9",
        }

        result = self.config.map_openai_params(
            image_edit_optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result.get("seed") == 42
        assert result.get("safety_tolerance") == 2
        assert result.get("aspect_ratio") == "16:9"
        assert result.get("output_format") == "png"

    def test_validate_environment_with_api_key(self):
        """Test environment validation with provided API key."""
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            api_key="test-api-key",
        )

        assert result["x-key"] == "test-api-key"
        assert result["Content-Type"] == "application/json"
        assert result["Accept"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """Test that missing API key raises error."""
        headers = {}

        with patch("litellm.llms.black_forest_labs.image_edit.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = None

            with pytest.raises(BlackForestLabsError) as exc_info:
                self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    api_key=None,
                )

            assert exc_info.value.status_code == 401
            assert "BFL_API_KEY is not set" in exc_info.value.message

    def test_get_model_endpoint_kontext_pro(self):
        """Test endpoint resolution for flux-kontext-pro."""
        endpoint = self.config._get_model_endpoint("flux-kontext-pro")
        assert endpoint == "/v1/flux-kontext-pro"

    def test_get_model_endpoint_kontext_max(self):
        """Test endpoint resolution for flux-kontext-max."""
        endpoint = self.config._get_model_endpoint("flux-kontext-max")
        assert endpoint == "/v1/flux-kontext-max"

    def test_get_model_endpoint_with_provider_prefix(self):
        """Test endpoint resolution with provider prefix."""
        endpoint = self.config._get_model_endpoint("black_forest_labs/flux-kontext-pro")
        assert endpoint == "/v1/flux-kontext-pro"

    def test_get_model_endpoint_fill(self):
        """Test endpoint resolution for flux-pro-1.0-fill."""
        endpoint = self.config._get_model_endpoint("flux-pro-1.0-fill")
        assert endpoint == "/v1/flux-pro-1.0-fill"

    def test_get_complete_url(self):
        """Test complete URL generation."""
        url = self.config.get_complete_url(
            model="flux-kontext-pro",
            api_base=None,
            litellm_params={},
        )

        assert url == "https://api.bfl.ai/v1/flux-kontext-pro"

    def test_get_complete_url_custom_base(self):
        """Test complete URL generation with custom base."""
        url = self.config.get_complete_url(
            model="flux-kontext-pro",
            api_base="https://custom.api.com/",
            litellm_params={},
        )

        assert url == "https://custom.api.com/v1/flux-kontext-pro"

    def test_transform_image_edit_request(self):
        """Test request transformation to BFL format."""
        image_data = b"fake_image_data"
        image = BytesIO(image_data)

        image_edit_optional_params = {
            "seed": 123,
            "output_format": "jpeg",
        }

        litellm_params = GenericLiteLLMParams()
        headers = {}

        data, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params=image_edit_optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Check that data contains the expected parameters
        assert data["prompt"] == self.prompt
        assert "input_image" in data
        # Verify base64 encoding
        decoded = base64.b64decode(data["input_image"])
        assert decoded == image_data
        assert data["seed"] == 123
        assert data["output_format"] == "jpeg"

        # BFL uses JSON, not multipart - files should be empty
        assert files == []

    def test_transform_image_edit_request_with_mask(self):
        """Test request transformation with mask for inpainting."""
        image_data = b"fake_image_data"
        mask_data = b"fake_mask_data"
        image = BytesIO(image_data)

        image_edit_optional_params = {
            "mask": BytesIO(mask_data),
            "output_format": "png",
        }

        litellm_params = GenericLiteLLMParams()
        headers = {}

        data, files = self.config.transform_image_edit_request(
            model="flux-pro-1.0-fill",
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params=image_edit_optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Check mask is base64 encoded
        assert "mask" in data
        decoded_mask = base64.b64decode(data["mask"])
        assert decoded_mask == mask_data

    def test_read_image_bytes_from_bytes(self):
        """Test reading image bytes from bytes input."""
        image_data = b"test_image_bytes"
        result = self.config._read_image_bytes(image_data)
        assert result == image_data

    def test_read_image_bytes_from_file_like(self):
        """Test reading image bytes from file-like object."""
        image_data = b"test_image_bytes"
        image = BytesIO(image_data)
        result = self.config._read_image_bytes(image)
        assert result == image_data

    def test_read_image_bytes_from_list(self):
        """Test reading image bytes from list (takes first)."""
        image_data = b"test_image_bytes"
        images = [BytesIO(image_data), BytesIO(b"other")]
        result = self.config._read_image_bytes(images)
        assert result == image_data

    def test_transform_image_edit_response_success(self):
        """Test response transformation with final polled response."""
        # The response is now the FINAL polled response from handler
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {"sample": "https://example.com/edited_image.png"},
        }
        mock_response.status_code = 200

        result = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/edited_image.png"

    def test_transform_image_edit_response_no_image_url(self):
        """Test response transformation when no image URL is present."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {},
        }
        mock_response.status_code = 200

        with pytest.raises(BlackForestLabsError, match="No image URL"):
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=self.logging_obj,
            )

    def test_transform_image_edit_response_json_parse_error(self):
        """Test response transformation with JSON parse error."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("error", "doc", 0)
        mock_response.status_code = 200

        with pytest.raises(BlackForestLabsError, match="Error parsing"):
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=self.logging_obj,
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

    def test_use_multipart_form_data_returns_false(self):
        """Test that use_multipart_form_data returns False for BFL."""
        assert self.config.use_multipart_form_data() is False


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("utf-8")


class TestFlux2ImageEdit:
    """
    FLUX.2 edits against up to 8 reference images (4 for [klein]) through the same
    endpoint it uses for text-to-image, so the references have to be spread over
    input_image/input_image_2../input_image_N instead of collapsing to the first one.
    """

    def setup_method(self):
        self.config = BlackForestLabsImageEditConfig()
        self.prompt = "Put the chair from the second photo into the first room"

    @pytest.mark.parametrize(
        "model,endpoint",
        [
            ("flux-2-max", "/v1/flux-2-max"),
            ("flux-2-pro", "/v1/flux-2-pro"),
            ("flux-2-pro-preview", "/v1/flux-2-pro-preview"),
            ("flux-2-flex", "/v1/flux-2-flex"),
            ("flux-2-klein-9b", "/v1/flux-2-klein-9b"),
            ("flux-2-klein-9b-preview", "/v1/flux-2-klein-9b-preview"),
            ("flux-2-klein-4b", "/v1/flux-2-klein-4b"),
        ],
    )
    def test_edit_uses_the_same_endpoint_as_generation(self, model, endpoint):
        assert (
            self.config.get_complete_url(model=f"black_forest_labs/{model}", api_base=None, litellm_params={})
            == f"https://api.bfl.ai{endpoint}"
        )

    def test_single_reference_uses_input_image(self):
        body, files = self.config.transform_image_edit_request(
            model="flux-2-pro",
            prompt=self.prompt,
            image=b"first",
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert files == []
        assert body["input_image"] == _b64(b"first")
        assert "input_image_2" not in body

    def test_multiple_references_are_numbered_from_two(self):
        body, _ = self.config.transform_image_edit_request(
            model="flux-2-max",
            prompt=self.prompt,
            image=[b"first", b"second", b"third"],
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert body["input_image"] == _b64(b"first")
        assert body["input_image_2"] == _b64(b"second")
        assert body["input_image_3"] == _b64(b"third")
        assert "input_image_4" not in body

    def test_eight_references_are_accepted_by_premium_variants(self):
        body, _ = self.config.transform_image_edit_request(
            model="flux-2-flex",
            prompt=self.prompt,
            image=[f"ref{index}".encode() for index in range(8)],
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert body["input_image"] == _b64(b"ref0")
        assert body["input_image_8"] == _b64(b"ref7")

    def test_ninth_reference_is_rejected(self):
        with pytest.raises(BlackForestLabsError, match="at most 8 reference"):
            self.config.transform_image_edit_request(
                model="flux-2-pro",
                prompt=self.prompt,
                image=[f"ref{index}".encode() for index in range(9)],
                image_edit_optional_request_params={},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    @pytest.mark.parametrize("model", ["flux-2-klein-9b", "flux-2-klein-9b-preview", "flux-2-klein-4b"])
    def test_klein_caps_references_at_four(self, model):
        with pytest.raises(BlackForestLabsError, match="at most 4 reference"):
            self.config.transform_image_edit_request(
                model=model,
                prompt=self.prompt,
                image=[f"ref{index}".encode() for index in range(5)],
                image_edit_optional_request_params={},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_empty_image_list_is_rejected(self):
        with pytest.raises(BlackForestLabsError, match="No image provided"):
            self.config.transform_image_edit_request(
                model="flux-2-pro",
                prompt=self.prompt,
                image=[],
                image_edit_optional_request_params={},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_mask_and_expand_params_are_not_supported(self):
        """FLUX.2 has no fill/expand surface, so those FLUX.1 params must not leak through."""
        params = self.config.get_supported_openai_params("flux-2-pro")

        for unsupported in ("mask", "grow_mask", "aspect_ratio", "top", "bottom", "left", "right"):
            assert unsupported not in params

    def test_fill_and_expand_params_never_reach_the_wire(self):
        """Even if a FLUX.1 param slips past validation it must not be sent to a FLUX.2 endpoint."""
        body, _ = self.config.transform_image_edit_request(
            model="flux-2-klein-4b",
            prompt=self.prompt,
            image=b"first",
            image_edit_optional_request_params={
                "mask": b"ignored",
                "aspect_ratio": "16:9",
                "grow_mask": 8,
                "top": 64,
                "steps": 20,
                "prompt_upsampling": True,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert body == {
            "prompt": self.prompt,
            "output_format": "png",
            "input_image": _b64(b"first"),
        }

    def test_map_openai_params_keeps_the_dimensions_it_advertises(self):
        """width/height are in tunable_params, so they must survive mapping instead of being dropped."""
        mapped = self.config.map_openai_params(
            ImageEditOptionalRequestParams(width=1440, height=2048),
            "flux-2-pro",
            drop_params=False,
        )

        assert mapped == {"output_format": "png", "width": 1440, "height": 2048}

    def test_advertised_edit_params_all_reach_the_wire(self):
        """Every param get_supported_openai_params promises has to survive mapping and transformation."""
        advertised = [param for param in self.config.get_supported_openai_params("flux-2-flex") if param != "steps"]
        sample = {"width": 1024, "height": 768, "seed": 7, "safety_tolerance": 2, "output_format": "jpeg"}
        sample.update({"guidance": 3.5, "prompt_upsampling": True})

        mapped = self.config.map_openai_params(sample, "flux-2-flex", drop_params=False)
        body, _ = self.config.transform_image_edit_request(
            model="flux-2-flex",
            prompt=self.prompt,
            image=b"first",
            image_edit_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert set(advertised) <= set(body)
        assert body["width"] == 1024
        assert body["height"] == 768
        assert body["output_format"] == "jpeg"

    def test_map_openai_params_drops_flux_1_only_params(self):
        """A FLUX.1 param that slips through must not survive mapping for a FLUX.2 model."""
        mapped = self.config.map_openai_params(
            {"aspect_ratio": "16:9", "grow_mask": 8, "top": 64, "seed": 11},
            "flux-2-pro",
            drop_params=False,
        )

        assert mapped == {"output_format": "png", "seed": 11}

    def test_map_openai_params_keeps_flex_step_control(self):
        mapped = self.config.map_openai_params(
            ImageEditOptionalRequestParams(steps=20, guidance=3.5, seed=11),
            "flux-2-flex",
            drop_params=False,
        )

        assert mapped == {"output_format": "png", "steps": 20, "guidance": 3.5, "seed": 11}

    def test_edit_body_inverts_prompt_upsampling_for_pro(self):
        body, _ = self.config.transform_image_edit_request(
            model="flux-2-pro",
            prompt=self.prompt,
            image=b"first",
            image_edit_optional_request_params={"prompt_upsampling": False, "output_format": "jpeg"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert body["disable_pup"] is True
        assert body["output_format"] == "jpeg"
        assert "prompt_upsampling" not in body

    def test_flux_1_still_collapses_a_list_to_the_first_image(self):
        """Regression guard: kontext has no multi-reference surface."""
        body, _ = self.config.transform_image_edit_request(
            model="flux-kontext-pro",
            prompt=self.prompt,
            image=[b"first", b"second"],
            image_edit_optional_request_params={"aspect_ratio": "1:1"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert body["input_image"] == _b64(b"first")
        assert "input_image_2" not in body
        assert body["aspect_ratio"] == "1:1"
