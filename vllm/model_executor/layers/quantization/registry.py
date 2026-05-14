# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass(frozen=True)
class QuantizationMethodSpec:
    """Static metadata used to resolve a quantization method.

    `config_cls_path` intentionally stays as a string for built-in methods so
    importing this module does not eagerly import every quantization backend.
    """

    name: str
    config_cls_path: str | None
    deprecated: bool = False
    override_priority: int | None = None
    config_cls: type[QuantizationConfig] | None = None

    def load_config_cls(self) -> type[QuantizationConfig]:
        if self.config_cls is not None:
            return self.config_cls
        if self.config_cls_path is None:
            raise ValueError(
                f"Quantization method {self.name!r} does not define a config class."
            )
        module_path, cls_name = self.config_cls_path.rsplit(".", 1)
        module = import_module(module_path)
        config_cls = getattr(module, cls_name)
        if not issubclass(config_cls, QuantizationConfig):
            raise TypeError(
                f"Quantization method {self.name!r} resolved to {config_cls}, "
                "which is not a QuantizationConfig subclass."
            )
        return config_cls


@dataclass(frozen=True)
class QuantizationResolution:
    """Result of resolving checkpoint and user quantization settings."""

    selected_method: str | None
    checkpoint_method: str | None
    override_method: str | None = None


# Keep this list in the same user-facing order as QuantizationMethods in
# __init__.py. The small duplication is retained so type checkers still see a
# Literal while runtime dispatch has a structured registry.
_BUILTIN_SPECS: tuple[QuantizationMethodSpec, ...] = (
    QuantizationMethodSpec(
        "awq", "vllm.model_executor.layers.quantization.awq.AWQConfig"
    ),
    QuantizationMethodSpec(
        "fp8", "vllm.model_executor.layers.quantization.fp8.Fp8Config"
    ),
    QuantizationMethodSpec(
        "fbgemm_fp8",
        "vllm.model_executor.layers.quantization.fbgemm_fp8.FBGEMMFp8Config",
        deprecated=True,
    ),
    QuantizationMethodSpec(
        "fp_quant",
        "vllm.model_executor.layers.quantization.fp_quant.FPQuantConfig",
        deprecated=True,
    ),
    QuantizationMethodSpec(
        "modelopt",
        "vllm.model_executor.layers.quantization.modelopt.ModelOptFp8Config",
        override_priority=40,
    ),
    QuantizationMethodSpec(
        "modelopt_fp4",
        "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4Config",
        override_priority=50,
    ),
    QuantizationMethodSpec(
        "modelopt_mxfp8",
        "vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8Config",
        override_priority=60,
    ),
    QuantizationMethodSpec(
        "modelopt_mixed",
        "vllm.model_executor.layers.quantization.modelopt.ModelOptMixedPrecisionConfig",
        override_priority=70,
    ),
    QuantizationMethodSpec(
        "gguf",
        "vllm.model_executor.layers.quantization.gguf.GGUFConfig",
        override_priority=130,
    ),
    QuantizationMethodSpec(
        "gptq_marlin",
        "vllm.model_executor.layers.quantization.gptq_marlin.GPTQMarlinConfig",
        override_priority=0,
    ),
    QuantizationMethodSpec(
        "awq_marlin",
        "vllm.model_executor.layers.quantization.awq_marlin.AWQMarlinConfig",
        override_priority=10,
    ),
    QuantizationMethodSpec(
        "gptq", "vllm.model_executor.layers.quantization.gptq.GPTQConfig"
    ),
    QuantizationMethodSpec(
        "humming",
        "vllm.model_executor.layers.quantization.humming.HummingConfig",
        override_priority=120,
    ),
    QuantizationMethodSpec(
        "compressed-tensors",
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors.CompressedTensorsConfig",
    ),
    QuantizationMethodSpec(
        "bitsandbytes",
        "vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig",
    ),
    QuantizationMethodSpec(
        "experts_int8",
        "vllm.model_executor.layers.quantization.experts_int8.ExpertsInt8Config",
    ),
    QuantizationMethodSpec(
        "quark",
        "vllm.model_executor.layers.quantization.quark.quark.QuarkConfig",
    ),
    QuantizationMethodSpec(
        "moe_wna16",
        "vllm.model_executor.layers.quantization.moe_wna16.MoeWNA16Config",
        override_priority=30,
    ),
    QuantizationMethodSpec(
        "torchao", "vllm.model_executor.layers.quantization.torchao.TorchAOConfig"
    ),
    QuantizationMethodSpec(
        "inc",
        "vllm.model_executor.layers.quantization.inc.INCConfig",
        override_priority=20,
    ),
    QuantizationMethodSpec(
        "mxfp4",
        "vllm.model_executor.layers.quantization.mxfp4.Mxfp4Config",
        override_priority=80,
    ),
    QuantizationMethodSpec(
        "gpt_oss_mxfp4",
        "vllm.model_executor.layers.quantization.mxfp4.GptOssMxfp4Config",
        override_priority=90,
    ),
    QuantizationMethodSpec(
        "deepseek_v4_fp8",
        "vllm.model_executor.models.deepseek_v4.DeepseekV4FP8Config",
        override_priority=100,
    ),
    QuantizationMethodSpec(
        "cpu_awq",
        "vllm.model_executor.layers.quantization.cpu_wna16.CPUAWQConfig",
        override_priority=110,
    ),
    QuantizationMethodSpec(
        "online",
        "vllm.model_executor.layers.quantization.online.base.OnlineQuantizationConfig",
    ),
    # Online quant shorthand names. These all resolve to OnlineQuantizationConfig.
    QuantizationMethodSpec(
        "fp8_per_tensor",
        "vllm.model_executor.layers.quantization.online.base.OnlineQuantizationConfig",
    ),
    QuantizationMethodSpec(
        "fp8_per_block",
        "vllm.model_executor.layers.quantization.online.base.OnlineQuantizationConfig",
    ),
    QuantizationMethodSpec(
        "int8_per_channel_weight_only",
        "vllm.model_executor.layers.quantization.online.base.OnlineQuantizationConfig",
    ),
    QuantizationMethodSpec(
        "mxfp8",
        "vllm.model_executor.layers.quantization.online.base.OnlineQuantizationConfig",
    ),
)

_BUILTIN_METHOD_TO_SPEC = {spec.name: spec for spec in _BUILTIN_SPECS}

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG: dict[str, type[QuantizationConfig]] = {}

QUANTIZATION_METHODS: list[str] = [spec.name for spec in _BUILTIN_SPECS]
DEPRECATED_QUANTIZATION_METHODS: list[str] = [
    spec.name for spec in _BUILTIN_SPECS if spec.deprecated
]


def get_quantization_method_specs() -> tuple[QuantizationMethodSpec, ...]:
    """Return static specs for built-in quantization methods."""
    return _BUILTIN_SPECS


def get_quantization_method_spec(quantization: str) -> QuantizationMethodSpec:
    try:
        return _BUILTIN_METHOD_TO_SPEC[quantization]
    except KeyError:
        if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
            return QuantizationMethodSpec(
                quantization,
                None,
                config_cls=_CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization],
            )
        raise ValueError(_unknown_quantization_message(quantization)) from None


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a
    customized quantization config to support it.
    """

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(_unknown_quantization_message(quantization))
    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]
    return _BUILTIN_METHOD_TO_SPEC[quantization].load_config_cls()


def is_deprecated_quantization_method(quantization: str | None) -> bool:
    return quantization in DEPRECATED_QUANTIZATION_METHODS


def _iter_override_candidate_names(user_quant: str | None) -> list[str]:
    candidates: list[str] = []

    builtin_override_specs = sorted(
        (spec for spec in _BUILTIN_SPECS if spec.override_priority is not None),
        key=lambda spec: (
            spec.override_priority if spec.override_priority is not None else 0
        ),
    )
    builtin_override_names = {spec.name for spec in builtin_override_specs}

    # Preserve the existing custom-first behavior for names that were not
    # already ordered as built-in overrides. Custom configs that overwrite an
    # existing built-in override keep that built-in priority slot below because
    # get_quantization_config() prefers the custom config class.
    candidates.extend(
        name
        for name in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG
        if name not in builtin_override_names
    )

    if user_quant == "humming":
        candidates.append("humming")

    candidates.extend(spec.name for spec in builtin_override_specs)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def resolve_quantization_method(
    hf_quant_cfg: dict[str, Any],
    user_quant: str | None,
    hf_config: Any = None,
) -> QuantizationResolution:
    """Resolve the quantization method from checkpoint metadata and user input."""

    if "quant_method" not in hf_quant_cfg:
        raise ValueError(
            "Quantization config is missing required key 'quant_method'. "
            f"Found keys: {sorted(hf_quant_cfg)}."
        )

    checkpoint_method = hf_quant_cfg["quant_method"]

    for name in _iter_override_candidate_names(user_quant):
        method = get_quantization_config(name)
        quantization_override = method.override_quantization_method(
            hf_quant_cfg, user_quant, hf_config=hf_config
        )
        if quantization_override is not None:
            return QuantizationResolution(
                selected_method=quantization_override,
                checkpoint_method=checkpoint_method,
                override_method=name,
            )

    checkpoint_method = checkpoint_method if checkpoint_method != "" else None
    if user_quant is None:
        return QuantizationResolution(
            selected_method=checkpoint_method,
            checkpoint_method=checkpoint_method,
        )

    if user_quant != checkpoint_method:
        raise ValueError(
            "Quantization method specified in the model config "
            f"({checkpoint_method}) does not match the quantization "
            f"method specified in the `quantization` argument ({user_quant})."
        )

    return QuantizationResolution(
        selected_method=user_quant,
        checkpoint_method=checkpoint_method,
    )


def validate_quantization_method(quantization: str) -> None:
    """Validate a selected quantization method against vLLM and platform support."""

    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(_unknown_quantization_message(quantization))

    try:
        current_platform.verify_quantization(quantization)
    except ValueError as exc:
        supported = current_platform.supported_quantization or QUANTIZATION_METHODS
        raise ValueError(
            f"Quantization method {quantization!r} is not supported on "
            f"{current_platform.device_name}. Supported quantization methods "
            f"for this platform: {supported}."
        ) from exc


def _unknown_quantization_message(quantization: str) -> str:
    return (
        f"Unknown quantization method: {quantization}. "
        f"Must be one of {QUANTIZATION_METHODS}."
    )
