"""
Black Forest Labs Common Utilities

Common utilities, constants, and error handling for Black Forest Labs API.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlparse

from typing_extensions import assert_never

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams


class BlackForestLabsError(BaseLLMException):
    """Exception class for Black Forest Labs API errors."""


# API Constants
DEFAULT_API_BASE: Final = "https://api.bfl.ai"

DEFAULT_OUTPUT_FORMAT: Final = "png"

# BFL uses regional subdomains (e.g. gateway.bfl.ai) for polling URLs that
# differ from the submission host (api.bfl.ai). We validate against the
# registered domain rather than doing a strict same-origin check.
_BFL_REGISTERED_DOMAIN: Final = "bfl.ai"


def assert_bfl_polling_url(polling_url: str) -> None:
    """Validate that a polling URL points to a BFL-controlled host.

    BFL returns polling URLs on subdomains like ``gateway.bfl.ai`` that differ
    from the submission host ``api.bfl.ai``. A strict same-origin check would
    reject these legitimate URLs. Instead we verify the host is ``bfl.ai`` or
    any subdomain of it, which keeps the SSRF guarantee (credentials only go
    to BFL-controlled infrastructure) without false-positives on regional hosts.

    Raises:
        BlackForestLabsError: If the polling URL scheme or host is not trusted.
    """
    parsed: Final = urlparse(polling_url)
    host: Final = (parsed.hostname or "").lower()

    if parsed.scheme != "https":
        raise BlackForestLabsError(
            status_code=502,
            message="Rejected polling URL: scheme must be https",
        )

    if host != _BFL_REGISTERED_DOMAIN and not host.endswith("." + _BFL_REGISTERED_DOMAIN):
        raise BlackForestLabsError(
            status_code=502,
            message="Rejected polling URL: host is not within the bfl.ai domain",
        )


# Polling configuration
DEFAULT_POLLING_INTERVAL: Final = 1.5  # seconds
DEFAULT_MAX_POLLING_TIME: Final = 300  # 5 minutes


PromptUpsamplingField = Literal["prompt_upsampling", "disable_pup"]

_FLUX_2_COMMON_REQUEST_PARAMS: Final[frozenset[OpenAIImageGenerationOptionalParams]] = frozenset(
    {"width", "height", "seed", "safety_tolerance", "output_format"}
)
_FLUX_2_STEP_CONTROL_REQUEST_PARAMS: Final[frozenset[OpenAIImageGenerationOptionalParams]] = frozenset(
    {"guidance", "steps"}
)
_FLUX_2_UPSAMPLING_PARAM: Final[frozenset[OpenAIImageGenerationOptionalParams]] = frozenset({"prompt_upsampling"})


@dataclass(frozen=True, slots=True)
class Flux2ModelSpec:
    """Endpoint and parameter surface of a single FLUX.2 model."""

    endpoint: str
    max_reference_images: int
    prompt_upsampling_field: PromptUpsamplingField | None
    supports_step_control: bool

    @property
    def request_params(self) -> frozenset[OpenAIImageGenerationOptionalParams]:
        """Wire field names accepted besides ``prompt``, the references and upsampling."""
        if self.supports_step_control:
            return _FLUX_2_COMMON_REQUEST_PARAMS | _FLUX_2_STEP_CONTROL_REQUEST_PARAMS
        return _FLUX_2_COMMON_REQUEST_PARAMS

    @property
    def tunable_params(self) -> frozenset[OpenAIImageGenerationOptionalParams]:
        """Knobs a caller may set: the wire fields plus the canonical ``prompt_upsampling`` flag."""
        if self.prompt_upsampling_field is None:
            return self.request_params
        return self.request_params | _FLUX_2_UPSAMPLING_PARAM


FLUX_2_MODELS: Final[Mapping[str, Flux2ModelSpec]] = MappingProxyType(
    {
        "flux-2-max": Flux2ModelSpec(
            endpoint="/v1/flux-2-max",
            max_reference_images=8,
            prompt_upsampling_field="disable_pup",
            supports_step_control=False,
        ),
        "flux-2-pro": Flux2ModelSpec(
            endpoint="/v1/flux-2-pro",
            max_reference_images=8,
            prompt_upsampling_field="disable_pup",
            supports_step_control=False,
        ),
        "flux-2-pro-preview": Flux2ModelSpec(
            endpoint="/v1/flux-2-pro-preview",
            max_reference_images=8,
            prompt_upsampling_field="disable_pup",
            supports_step_control=False,
        ),
        "flux-2-flex": Flux2ModelSpec(
            endpoint="/v1/flux-2-flex",
            max_reference_images=8,
            prompt_upsampling_field="prompt_upsampling",
            supports_step_control=True,
        ),
        "flux-2-klein-9b": Flux2ModelSpec(
            endpoint="/v1/flux-2-klein-9b",
            max_reference_images=4,
            prompt_upsampling_field=None,
            supports_step_control=False,
        ),
        "flux-2-klein-9b-preview": Flux2ModelSpec(
            endpoint="/v1/flux-2-klein-9b-preview",
            max_reference_images=4,
            prompt_upsampling_field=None,
            supports_step_control=False,
        ),
        "flux-2-klein-4b": Flux2ModelSpec(
            endpoint="/v1/flux-2-klein-4b",
            max_reference_images=4,
            prompt_upsampling_field=None,
            supports_step_control=False,
        ),
    }
)

_FLUX_2_ENDPOINTS: Final[Mapping[str, str]] = MappingProxyType(
    {model: spec.endpoint for model, spec in FLUX_2_MODELS.items()}
)


def strip_provider_prefix(model: str) -> str:
    """Normalize ``black_forest_labs/flux-2-pro`` (or a bare model name) to ``flux-2-pro``."""
    return model.lower().rsplit("/", 1)[-1]


def get_flux_2_model_spec(model: str) -> Flux2ModelSpec | None:
    """Return the FLUX.2 spec for ``model``, or ``None`` when it is not a FLUX.2 model."""
    return FLUX_2_MODELS.get(strip_provider_prefix(model))


def flux_2_reference_image_field(index: int) -> str:
    """Wire field name for the ``index``-th (0-based) reference image."""
    return "input_image" if index == 0 else f"input_image_{index + 1}"


def build_flux_2_request_body(
    spec: Flux2ModelSpec,
    prompt: str,
    optional_params: Mapping[str, object],
    extra_fields: tuple[tuple[str, object], ...] = (),
) -> dict[str, object]:
    """Build the JSON body a FLUX.2 endpoint expects, dropping fields it does not accept.

    ``extra_fields`` carries already-encoded fields the caller owns, such as the
    ``input_image``/``input_image_N`` references of an edit request.
    """
    passthrough: Final = tuple(
        (key, value) for key, value in optional_params.items() if key in spec.request_params and value is not None
    )
    return dict(  # mutable-ok: the BaseImageGenerationConfig/BaseImageEditConfig contract returns a real dict
        (
            ("prompt", prompt),
            ("output_format", DEFAULT_OUTPUT_FORMAT),
            *passthrough,
            *_flux_2_prompt_upsampling_entry(spec, optional_params.get("prompt_upsampling")),
            *extra_fields,
        )
    )


def _flux_2_prompt_upsampling_entry(
    spec: Flux2ModelSpec,
    prompt_upsampling: object,
) -> tuple[tuple[str, bool], ...]:
    if not isinstance(prompt_upsampling, bool):
        return ()

    match spec.prompt_upsampling_field:
        case "prompt_upsampling":
            return (("prompt_upsampling", prompt_upsampling),)
        case "disable_pup":
            return (("disable_pup", not prompt_upsampling),)
        case None:
            return ()
        case _ as unreachable:
            assert_never(unreachable)


# Model to endpoint mapping for image edit
IMAGE_EDIT_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "flux-kontext-pro": "/v1/flux-kontext-pro",
        "flux-kontext-max": "/v1/flux-kontext-max",
        "flux-pro-1.0-fill": "/v1/flux-pro-1.0-fill",
        "flux-pro-1.0-expand": "/v1/flux-pro-1.0-expand",
        **_FLUX_2_ENDPOINTS,
    }
)

# Model to endpoint mapping for image generation
IMAGE_GENERATION_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "flux-pro-1.1": "/v1/flux-pro-1.1",
        "flux-pro-1.1-ultra": "/v1/flux-pro-1.1-ultra",
        "flux-dev": "/v1/flux-dev",
        "flux-pro": "/v1/flux-pro",
        # Kontext models support both text-to-image and image editing
        "flux-kontext-pro": "/v1/flux-kontext-pro",
        "flux-kontext-max": "/v1/flux-kontext-max",
        **_FLUX_2_ENDPOINTS,
    }
)
