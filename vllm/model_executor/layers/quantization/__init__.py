# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.registry import (
    DEPRECATED_QUANTIZATION_METHODS,
    QUANTIZATION_METHODS,
    QuantizationMethodSpec,
    QuantizationResolution,
    get_quantization_config,
    get_quantization_method_spec,
    get_quantization_method_specs,
    is_deprecated_quantization_method,
    register_quantization_config,
    resolve_quantization_method,
    validate_quantization_method,
)

QuantizationMethods = Literal[
    "awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "gguf",
    "gptq_marlin",
    "awq_marlin",
    "gptq",
    "humming",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "cpu_awq",
    "online",
    # Below are online quant shorthand names (see vllm.config.quantization).
    # Listed here as strings to avoid a circular import; kept in sync with the
    # runtime registry by the assertion below.
    "fp8_per_tensor",
    "fp8_per_block",
    "int8_per_channel_weight_only",
    "mxfp8",
]
assert set(QUANTIZATION_METHODS) == set(get_args(QuantizationMethods)), (
    "QuantizationMethods and the quantization registry are out of sync."
)


__all__ = [
    "QuantizationConfig",
    "QuantizationMethodSpec",
    "QuantizationMethods",
    "QuantizationResolution",
    "get_quantization_method_spec",
    "get_quantization_method_specs",
    "get_quantization_config",
    "is_deprecated_quantization_method",
    "register_quantization_config",
    "resolve_quantization_method",
    "validate_quantization_method",
    "QUANTIZATION_METHODS",
    "DEPRECATED_QUANTIZATION_METHODS",
]
