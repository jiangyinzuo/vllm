# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerNameType,
    _encode_layer_name,
)

from ..vllm_inductor_pass import VllmFusionPatternMatcherPass, VllmPatternReplacement
from .matcher_utils import MatcherDeepseekScalingRotaryEmbedding, MatcherRotaryEmbedding

logger = init_logger(__name__)


class MLARoPEKVCacheCatPattern(VllmPatternReplacement):
    FUSED_OP = torch.ops.vllm.fused_rope_unified_mla_kv_cache_update.default

    def __init__(
        self,
        layer: MLAAttention,
        is_neox: bool,
        use_flashinfer: bool = False,
        use_deepseek_scaling: bool = False,
    ) -> None:
        self.layer_name = layer.layer_name
        self.kv_cache_dtype = layer.kv_cache_dtype
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        self.is_neox = is_neox
        self.use_flashinfer = use_flashinfer
        self._ln = _encode_layer_name(self.layer_name)

        if use_deepseek_scaling:
            self.rope_matcher = MatcherDeepseekScalingRotaryEmbedding(
                is_neox=self.is_neox,
                head_size=self.qk_rope_head_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                use_flashinfer=self.use_flashinfer,
            )
        else:
            self.rope_matcher = MatcherRotaryEmbedding(  # type: ignore
                is_neox=self.is_neox,
                head_size=self.qk_rope_head_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                use_flashinfer=self.use_flashinfer,
            )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        q_pe = self.empty_bf16(T, self.num_heads, self.qk_rope_head_dim)
        k_pe = self.empty_bf16(T, self.qk_rope_head_dim)
        kv_c_normed = self.empty_bf16(T, self.kv_lora_rank)
        cos_sin_cache = self.empty_bf16(L, self.qk_rope_head_dim)
        positions = self.empty(T, dtype=torch.int64)
        k_scale = self.empty(0, dtype=torch.float32)
        inputs = [
            q_pe,
            k_pe,
            kv_c_normed,
            positions,
            cos_sin_cache,
            k_scale,
        ]
        if _USE_LAYERNAME:
            inputs.append(self._ln)
        return inputs

    @property
    def pattern(self):
        _ln = self._ln

        if _USE_LAYERNAME:

            def _pattern_with_ln(
                q_pe: torch.Tensor,
                k_pe: torch.Tensor,
                kv_c_normed: torch.Tensor,
                positions: torch.Tensor,
                cos_sin_cache: torch.Tensor,
                k_scale: torch.Tensor,
                layer_name: LayerNameType,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                k_pe_unsqueezed = k_pe.unsqueeze(1)
                q_pe, k_pe = self.rope_matcher(
                    positions, q_pe, k_pe_unsqueezed, cos_sin_cache
                )
                dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                    kv_c_normed, k_pe, layer_name, self.kv_cache_dtype, k_scale
                )
                return dummy, q_pe, k_pe

            return _pattern_with_ln

        def _pattern(
            q_pe: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            k_pe_unsqueezed = k_pe.unsqueeze(1)
            q_pe, k_pe = self.rope_matcher(
                positions, q_pe, k_pe_unsqueezed, cos_sin_cache
            )
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, k_pe, _ln, self.kv_cache_dtype, k_scale
            )
            return dummy, q_pe, k_pe

        return _pattern

    @property
    def replacement(self):
        _ln = self._ln

        if _USE_LAYERNAME:

            def _replacement_with_ln(
                q_pe: torch.Tensor,
                k_pe: torch.Tensor,
                kv_c_normed: torch.Tensor,
                positions: torch.Tensor,
                cos_sin_cache: torch.Tensor,
                k_scale: torch.Tensor,
                layer_name: LayerNameType,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                at = auto_functionalized(
                    self.FUSED_OP,
                    positions=positions,
                    q_pe=q_pe,
                    k_pe=k_pe,
                    kv_c=kv_c_normed,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=self.is_neox,
                    kv_cache_dtype=self.kv_cache_dtype,
                    kv_cache_scale=k_scale,
                    layer_name=layer_name,
                )
                dummy, q_pe, k_pe_squeezed = at
                k_pe = k_pe_squeezed.unsqueeze(1)
                return dummy, q_pe, k_pe

            return _replacement_with_ln

        def _replacement(
            q_pe: torch.Tensor,
            k_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = auto_functionalized(
                self.FUSED_OP,
                positions=positions,
                q_pe=q_pe,
                k_pe=k_pe,
                kv_c=kv_c_normed,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                kv_cache_dtype=self.kv_cache_dtype,
                kv_cache_scale=k_scale,
                layer_name=_ln,
            )
            dummy, q_pe, k_pe_squeezed = at
            k_pe = k_pe_squeezed.unsqueeze(1)
            return dummy, q_pe, k_pe

        return _replacement


class MLARoPEKVCacheCatFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "mla_rope_kv_cache_fusion_pass")

        attn_layers = get_layers_from_vllm_config(config, MLAAttention)

        for _, layer in attn_layers.items():
            for is_neox in [False, True]:
                for use_deepseek_scaling in [False, True]:
                    if RotaryEmbedding.enabled():
                        for use_flashinfer in [False, True]:
                            self.register(
                                MLARoPEKVCacheCatPattern(
                                    layer,
                                    is_neox,
                                    use_flashinfer,
                                    use_deepseek_scaling,
                                )
                            )
                    else:
                        self.register(
                            MLARoPEKVCacheCatPattern(
                                layer,
                                is_neox,
                                use_deepseek_scaling=use_deepseek_scaling,
                            )
                        )

            if _USE_LAYERNAME:
                break

        self.dump_patterns(config, self.pm_pass)
