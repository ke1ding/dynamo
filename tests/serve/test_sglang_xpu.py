# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    chat_payload,
    chat_payload_default,
    chat_payload_with_logprobs,
    completion_payload_default,
    completion_payload_with_logprobs,
    guided_decoding_chat_payload_default,
    kv_events_metrics_payload,
    metric_payload_default,
    responses_payload_default,
    responses_stream_payload_default,
    router_selection_chat_payload_default,
)
from tests.utils.payloads import ResponsesPayload, ResponsesStreamPayload

logger = logging.getLogger(__name__)


def _disable_responses_reasoning(
    payload: ResponsesPayload | ResponsesStreamPayload,
) -> ResponsesPayload | ResponsesStreamPayload:
    # Keep the streaming and non-streaming smoke tests consistent by preventing
    # reasoning from consuming the entire output token budget.
    payload.body["reasoning"] = {"effort": "none"}
    return payload


@dataclass
class SGLangConfig(EngineConfig):
    """Configuration for SGLang test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["SGLANG:EngineCore"])


sglang_dir = os.environ.get("SGLANG_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/sglang"
)


@pytest.fixture(autouse=True)
def clear_stale_fpm_env(monkeypatch):
    """Keep serve/XPU tests isolated from stale FPM env vars inherited by CI or parent shells."""
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.delenv("DYN_FPM_TRACE", raising=False)


# SGLang test configurations
sglang_configs = {
    "aggregated": SGLangConfig(
        # Uses backend agg.sh (with metrics enabled) for testing standard
        # aggregated deployment with metrics collection
        name="aggregated_xpu",
        directory=sglang_dir,
        script_name="xpu/agg_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
            pytest.mark.profiled_vram_gib(
                3.7
            ),  # ~2.9 GiB peak on Arc Pro B70 at recommended token count
            pytest.mark.requested_sglang_kv_tokens(
                2048
            ),  # >= prompt(~16) + max_tokens(1000) + scheduler reserve;
            # SGLang 0.5.11 silently hangs (no scheduler activity, no error)
            # when prompt+max_tokens nears max_total_tokens. Bisected hang
            # threshold ~1040 for these payloads; 2048 leaves headroom.
            pytest.mark.timeout(360),  # ~4x ~88s on Arc Pro B70
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            chat_payload(
                "Name one color in a short sentence.",
                repeat_count=1,
                expected_response=[],
                max_tokens=16,
                extra_body={"n": 2},
                expected_num_choices=2,
            ),
            completion_payload_default(),
            _disable_responses_reasoning(responses_payload_default()),
            _disable_responses_reasoning(responses_stream_payload_default()),
            guided_decoding_chat_payload_default(),
            metric_payload_default(min_num_requests=6, backend="sglang"),
        ],
    ),
    # Top-k logprobs on chat and completions with no opt-in env var; mirrors
    # the vLLM and TensorRT-LLM aggregated_logprobs configs.
    "aggregated_logprobs": SGLangConfig(
        name="aggregated_logprobs_xpu",
        directory=sglang_dir,
        script_name="xpu/agg_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
            pytest.mark.profiled_vram_gib(
                3.9
            ),  # ~2.3 GiB peak on Arc Pro B70 at recommended token count
            pytest.mark.requested_sglang_kv_tokens(
                512
            ),  # 2x safety over min=256; payloads are ~16-token prompts + 30 max_tokens
            pytest.mark.timeout(360),  # ~12x observed ~30s on Arc Pro B70
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_with_logprobs(
                repeat_count=2,
                expected_response=["AI", "knock", "joke"],
                max_tokens=30,
                temperature=0.0,
                top_logprobs=3,
            ),
            completion_payload_with_logprobs(
                repeat_count=2,
                expected_response=["AI", "knock", "joke"],
                max_tokens=30,
                temperature=0.0,
                logprobs=5,
            ),
        ],
    ),
    # Speculative decoding: Qwen3-8B main model with an EAGLE3 draft model
    # (see launch/agg_spec_decoding.sh). Both repos are ungated.
    # Nightly-only: the 8B base plus EAGLE3 draft model is intentionally outside pre-merge CI.
    "kv_events": SGLangConfig(
        name="kv_events_xpu",
        directory=sglang_dir,
        script_name="xpu/agg_router_xpu.sh",
        marks=[
            pytest.mark.router,
            pytest.mark.xpu_2,
            pytest.mark.profiled_vram_gib(
                7.4
            ),  # 2x 3.7 GiB (one worker per XPU); B70 peak ~2.6 + ~2.6 GiB
            pytest.mark.requested_sglang_kv_tokens(
                2048
            ),  # per worker; same budget as "aggregated" (max_tokens=1000 payloads)
            pytest.mark.timeout(240),  # ~4x ~53s on 2x Arc Pro B70
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            router_selection_chat_payload_default(),
            kv_events_metrics_payload(system_ports=[DefaultPort.SYSTEM2.value]),
        ],
    ),
    "completions_only": SGLangConfig(
        name="completions_only_xpu",
        directory=sglang_dir,
        script_name="xpu/agg_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
            # Verifies dynamo+backend can serve a model that ships NO chat
            # template, via the completions endpoint. The model is NOT
            # incidental: it must be a base model without a chat template.
            # TinyLlama-1.1B (intermediate base checkpoint) is a small Llama-
            # family base without a chat template (replaces deepseek-llm-7b-base,
            # 7B) -- keeps the coverage, cuts VRAM. TinyLlama-1.1B-Chat is
            # already used in the router e2e suite.
            # Profiled on an Arc Pro B70 at the 2048-token KV cap below.
            pytest.mark.profiled_vram_gib(3.9),  # ~3.3 GiB peak
            pytest.mark.requested_sglang_kv_tokens(
                2048
            ),  # >= prompt(~16) + max_tokens(1000) + scheduler reserve;
            # SGLang 0.5.11 silently hangs when prompt+max_tokens nears
            # max_total_tokens (bisected ~1040 for these payloads). Matches
            # the "aggregated" config above.
            pytest.mark.timeout(300),  # 1.1B loads quickly; CI margin
            pytest.mark.post_merge,
        ],
        model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        script_args=[
            "--model-path",
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "--dyn-endpoint-types",
            "completions",
        ],
        request_payloads=[
            completion_payload_default(),
        ],
    ),
    # ── Diffusion pre_merge smoke tests ─────────────────────────────────
}


@pytest.fixture(params=params_with_model_mark(sglang_configs))
def sglang_config_test(request):
    """Fixture that provides different SGLang test configurations"""
    return sglang_configs[request.param]


@pytest.mark.e2e
@pytest.mark.sglang
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_sglang_deployment(
    sglang_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Test SGLang deployment scenarios using common helpers"""
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    config = dataclasses.replace(
        sglang_config_test,
        frontend_port=dynamo_dynamic_ports.frontend_port,
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
