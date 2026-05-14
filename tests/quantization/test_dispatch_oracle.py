# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for quantization method registry and resolution.

These tests exercise config-level dispatch only. They must not instantiate an
engine, load model weights, or run vLLM serving.
"""

from typing import get_args

import pytest

from vllm.config.quantization import _ONLINE_SHORTHANDS
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    QuantizationMethods,
    get_quantization_config,
    get_quantization_method_specs,
    is_deprecated_quantization_method,
    registry,
    resolve_quantization_method,
    validate_quantization_method,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig


def test_registry_matches_public_literal() -> None:
    assert {spec.name for spec in get_quantization_method_specs()} == set(
        get_args(QuantizationMethods)
    )
    assert set(get_args(QuantizationMethods)).issubset(QUANTIZATION_METHODS)


def test_get_quantization_config_loads_builtin_config_class() -> None:
    quant_config_cls = get_quantization_config("awq")
    assert quant_config_cls is AWQConfig
    assert issubclass(quant_config_cls, QuantizationConfig)


@pytest.mark.parametrize("shorthand", sorted(_ONLINE_SHORTHANDS))
def test_online_shorthands_resolve_to_online_config(shorthand: str) -> None:
    assert shorthand in QUANTIZATION_METHODS
    assert get_quantization_config(shorthand) is OnlineQuantizationConfig


def test_deprecated_methods_are_declared_in_registry() -> None:
    assert is_deprecated_quantization_method("fbgemm_fp8")
    assert is_deprecated_quantization_method("fp_quant")
    assert not is_deprecated_quantization_method("awq")


def test_resolve_quantization_method_uses_override(monkeypatch) -> None:
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        assert hf_quant_cfg["quant_method"] == "gptq"
        assert user_quant is None
        return "gptq_marlin"

    monkeypatch.setattr(
        GPTQMarlinConfig,
        "override_quantization_method",
        classmethod(override_quantization_method),
    )

    resolution = resolve_quantization_method({"quant_method": "gptq"}, None)

    assert resolution.selected_method == "gptq_marlin"
    assert resolution.checkpoint_method == "gptq"
    assert resolution.override_method == "gptq_marlin"


def test_resolve_quantization_method_keeps_explicit_user_method(monkeypatch) -> None:
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        assert user_quant == "gptq"
        return None

    monkeypatch.setattr(
        GPTQMarlinConfig,
        "override_quantization_method",
        classmethod(override_quantization_method),
    )

    resolution = resolve_quantization_method({"quant_method": "gptq"}, "gptq")

    assert resolution.selected_method == "gptq"
    assert resolution.checkpoint_method == "gptq"
    assert resolution.override_method is None


def test_resolve_quantization_method_rejects_user_checkpoint_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        resolve_quantization_method({"quant_method": "fp8"}, "gptq")


def test_resolve_quantization_method_rejects_missing_quant_method() -> None:
    with pytest.raises(ValueError, match="missing required key 'quant_method'"):
        resolve_quantization_method({"bits": 4}, None)


def test_validate_quantization_method_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown quantization method: not_real"):
        validate_quantization_method("not_real")


def test_validate_quantization_method_reports_platform_support(monkeypatch) -> None:
    def verify_quantization(quantization: str) -> None:
        raise ValueError(f"{quantization} is unsupported")

    monkeypatch.setattr(
        registry.current_platform,
        "supported_quantization",
        ["awq"],
    )
    monkeypatch.setattr(
        registry.current_platform,
        "verify_quantization",
        verify_quantization,
    )
    with pytest.raises(ValueError, match="Supported quantization methods"):
        validate_quantization_method("gptq")
