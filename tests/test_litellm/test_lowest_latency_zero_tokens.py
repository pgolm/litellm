#### What this tests ####
#    This tests the router's handling of zero completion tokens in lowest latency routing

import json
import os
import sys
import time
from datetime import datetime, timedelta

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler


def test_zero_completion_tokens_no_division_error():
    """
    Test that log_success_event handles zero completion tokens without ZeroDivisionError

    This tests the fix for issue #12641 where responses with zero completion tokens
    (e.g., from Gemini with long contexts) caused ZeroDivisionError
    """
    test_cache = DualCache()

    lowest_latency_logger = LowestLatencyLoggingHandler(router_cache=test_cache)

    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gemini-2.5-flash",
                "deployment": "gemini/gemini-2.5-flash",
            },
            "model_info": {"id": deployment_id},
        }
    }

    # Create a ModelResponse with zero completion tokens (as reported in issue)
    response_obj = litellm.ModelResponse(
        id="9p13aIGDDNmPmLAP5-23mQQ",
        created=1752669685,
        model="gemini-2.5-flash",
        object="chat.completion",
        choices=[
            litellm.Choices(
                finish_reason="stop",
                index=0,
                message=litellm.Message(
                    content=None, role="assistant", tool_calls=None
                ),
            )
        ],
        usage=litellm.Usage(
            completion_tokens=0,  # This causes the ZeroDivisionError
            prompt_tokens=245537,
            total_tokens=245537,
        ),
    )

    start_time = time.time()
    time.sleep(0.1)  # Simulate some response time
    end_time = time.time()

    # This should not raise ZeroDivisionError
    try:
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    except ZeroDivisionError:
        pytest.fail(
            "log_success_event raised ZeroDivisionError with zero completion tokens"
        )

    # Verify the deployment was logged (even with zero completion tokens)
    cached_value = test_cache.get_cache(
        key=f"{kwargs['litellm_params']['metadata']['model_group']}_map"
    )
    assert cached_value is not None
    assert deployment_id in cached_value


def test_zero_completion_tokens_with_time_to_first_token():
    """
    Test that time_to_first_token calculation also handles zero completion tokens
    """
    test_cache = DualCache()

    lowest_latency_logger = LowestLatencyLoggingHandler(router_cache=test_cache)

    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gemini-2.5-flash",
                "deployment": "gemini/gemini-2.5-flash",
            },
            "model_info": {"id": deployment_id},
            "stream": True,
        },
        "completion_start_time": time.time() + 0.05,  # Simulate time to first token
    }

    # Create a ModelResponse with zero completion tokens
    response_obj = litellm.ModelResponse(
        usage=litellm.Usage(
            completion_tokens=0, prompt_tokens=100000, total_tokens=100000
        )
    )

    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()

    # This should not raise ZeroDivisionError
    try:
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    except ZeroDivisionError:
        pytest.fail(
            "log_success_event raised ZeroDivisionError with zero completion tokens in streaming"
        )


def _kwargs(model_group, deployment_id, stream=False, completion_start_time=None):
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": f"openai/{model_group}",
            },
            "model_info": {"id": deployment_id},
        }
    }
    if stream:
        kwargs["stream"] = True
    if completion_start_time is not None:
        kwargs["completion_start_time"] = completion_start_time
    return kwargs


def _latency_values(cache, model_group, deployment_id):
    cached = cache.get_cache(key=f"{model_group}_map")
    return cached[deployment_id]["latency"], cached


def test_embedding_response_latency_is_json_serializable():
    """
    Regression for #8025 / #14383: non-ModelResponse types (embedding, audio,
    image) used to leave response time as a datetime.timedelta, which crashed the
    Redis JSON write with "Object of type timedelta is not JSON serializable".

    The real logging path passes datetime objects, so end_time - start_time is a
    timedelta; this is what the float-based existing tests never exercised.
    """
    test_cache = DualCache()
    logger = LowestLatencyLoggingHandler(router_cache=test_cache)

    model_group = "text-embedding-3-small"
    deployment_id = "emb-1"

    response_obj = litellm.EmbeddingResponse(
        model=model_group,
        data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        usage=litellm.Usage(prompt_tokens=5, total_tokens=5),
    )

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=0.42)

    logger.log_success_event(
        response_obj=response_obj,
        kwargs=_kwargs(model_group, deployment_id),
        start_time=start_time,
        end_time=end_time,
    )

    latency_values, request_count_dict = _latency_values(
        test_cache, model_group, deployment_id
    )
    assert latency_values, "expected a latency value to be appended"
    assert all(isinstance(v, float) for v in latency_values)
    assert not any(isinstance(v, timedelta) for v in latency_values)
    # The actual crash: this would raise TypeError before the fix
    json.dumps(request_count_dict)


@pytest.mark.asyncio
async def test_async_embedding_response_latency_is_json_serializable():
    test_cache = DualCache()
    logger = LowestLatencyLoggingHandler(router_cache=test_cache)

    model_group = "text-embedding-3-small"
    deployment_id = "emb-async-1"

    response_obj = litellm.EmbeddingResponse(
        model=model_group,
        data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        usage=litellm.Usage(prompt_tokens=4, total_tokens=4),
    )

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=0.31)

    await logger.async_log_success_event(
        response_obj=response_obj,
        kwargs=_kwargs(model_group, deployment_id),
        start_time=start_time,
        end_time=end_time,
    )

    latency_values, request_count_dict = _latency_values(
        test_cache, model_group, deployment_id
    )
    assert latency_values
    assert all(isinstance(v, float) for v in latency_values)
    json.dumps(request_count_dict)


def test_chat_completion_latency_unchanged_with_datetime():
    """
    The chat path must keep dividing latency by completion tokens and still emit
    a float when start/end are datetime objects.
    """
    test_cache = DualCache()
    logger = LowestLatencyLoggingHandler(router_cache=test_cache)

    model_group = "gpt-4o-mini"
    deployment_id = "chat-1"

    response_obj = litellm.ModelResponse(
        model=model_group,
        usage=litellm.Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30),
    )

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=2.0)

    logger.log_success_event(
        response_obj=response_obj,
        kwargs=_kwargs(model_group, deployment_id),
        start_time=start_time,
        end_time=end_time,
    )

    latency_values, request_count_dict = _latency_values(
        test_cache, model_group, deployment_id
    )
    assert latency_values == [pytest.approx(2.0 / 10)]
    json.dumps(request_count_dict)


def test_streaming_time_to_first_token_is_float_with_datetime():
    test_cache = DualCache()
    logger = LowestLatencyLoggingHandler(router_cache=test_cache)

    model_group = "gpt-4o-mini"
    deployment_id = "chat-stream-1"

    response_obj = litellm.ModelResponse(
        model=model_group,
        usage=litellm.Usage(completion_tokens=8, prompt_tokens=12, total_tokens=20),
    )

    start_time = datetime.now()
    completion_start_time = start_time + timedelta(seconds=0.5)
    end_time = start_time + timedelta(seconds=1.6)

    logger.log_success_event(
        response_obj=response_obj,
        kwargs=_kwargs(
            model_group,
            deployment_id,
            stream=True,
            completion_start_time=completion_start_time,
        ),
        start_time=start_time,
        end_time=end_time,
    )

    cached = test_cache.get_cache(key=f"{model_group}_map")
    ttft_values = cached[deployment_id]["time_to_first_token"]
    assert ttft_values == [pytest.approx(0.5 / 8)]
    assert all(isinstance(v, float) for v in ttft_values)
    json.dumps(cached)


if __name__ == "__main__":
    test_zero_completion_tokens_no_division_error()
    test_zero_completion_tokens_with_time_to_first_token()
    print("All tests passed!")
