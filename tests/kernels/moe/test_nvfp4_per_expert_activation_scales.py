# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Numerical coverage for per-expert NVFP4 GEMM2 activation scales."""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import dequantize_nvfp4_to_dtype
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_cutlass_fused_moe,
    has_flashinfer_trtllm_fused_moe,
)


def _dequantize_weights(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_encode_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty(
        (*weight.shape[:-1], weight.shape[-1] * 2),
        device=weight.device,
        dtype=dtype,
    )
    for expert in range(weight.shape[0]):
        out[expert] = dequantize_nvfp4_to_dtype(
            weight[expert],
            block_scale[expert],
            global_encode_scale[expert],
            dtype=dtype,
            device=weight.device,
            block_size=16,
        )
    return out


def _reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
) -> torch.Tensor:
    """Run top-k=1 SwiGLU with the same two static NVFP4 QDQ steps."""
    dtype = hidden_states.dtype
    a1_q, a1_block_scale = ops.scaled_fp4_quant(hidden_states, a1_gscale[0])
    a1 = dequantize_nvfp4_to_dtype(
        a1_q,
        a1_block_scale,
        a1_gscale[0],
        dtype=dtype,
        device=hidden_states.device,
        block_size=16,
    )

    out = torch.empty_like(hidden_states)
    for row, expert in enumerate(topk_ids[:, 0].tolist()):
        intermediate = SiluAndMul()(a1[row : row + 1] @ w1[expert].T)
        a2_q, a2_block_scale = ops.scaled_fp4_quant(intermediate, a2_gscale[expert])
        a2 = dequantize_nvfp4_to_dtype(
            a2_q,
            a2_block_scale,
            a2_gscale[expert],
            dtype=dtype,
            device=hidden_states.device,
            block_size=16,
        )
        out[row] = a2 @ w2[expert].T
    return out


@pytest.mark.parametrize("backend", ["flashinfer_trtllm", "flashinfer_cutlass"])
@torch.inference_mode()
def test_nvfp4_per_expert_gemm2_activation_scales_numerics(
    backend: str, workspace_init
) -> None:
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        pytest.skip("NVFP4 MoE needs a Blackwell (SM100) GPU.")
    if backend == "flashinfer_trtllm" and not has_flashinfer_trtllm_fused_moe():
        pytest.skip("FlashInfer TRTLLM NVFP4 MoE is unavailable.")
    if backend == "flashinfer_cutlass" and not has_flashinfer_cutlass_fused_moe():
        pytest.skip("FlashInfer CUTLASS NVFP4 MoE is unavailable.")

    torch.manual_seed(7)
    m = e = 8
    n = k = 1024
    dtype = torch.bfloat16
    activation = MoEActivation.SILU

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=True,
            is_scale_swizzled=backend != "flashinfer_trtllm",
        )
        assert quant_config.g1_alphas is not None
        assert quant_config.g2_alphas is not None
        assert quant_config.a1_gscale is not None
        assert quant_config.a2_gscale is not None
        assert quant_config.w1_scale is not None
        assert quant_config.w2_scale is not None

        # The outer global encode scales deliberately span enough E4M3 bins to
        # make accidental pooling observable. Fold the matching activation
        # decode scale into GEMM2's output alpha, as production loading does.
        a2_gscale = torch.tensor(
            [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 3000.0, 30000.0],
            device="cuda",
            dtype=torch.float32,
        )
        w1_global_encode_scale = 1.0 / quant_config.g1_alphas.clone()
        w2_global_encode_scale = 1.0 / quant_config.g2_alphas.clone()
        quant_config.a2_gscale.copy_(a2_gscale)
        quant_config.g2_alphas.div_(a2_gscale)

        topk_ids = torch.arange(e, device="cuda", dtype=torch.int64).view(m, 1)
        topk_weights = torch.ones((m, 1), device="cuda", dtype=torch.float32)
        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=1,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=m,
        )

        if backend == "flashinfer_trtllm":
            experts = TrtLlmNvFp4ExpertsModular(moe_config, quant_config)
            fake_layer = torch.nn.Module()
            fake_layer.w13_weight_scale_2 = quant_config.g1_alphas
            fake_layer.w2_weight_scale_2 = quant_config.g2_alphas
            fake_layer.w13_input_scale = torch.ones_like(quant_config.g1_alphas)
            fake_layer.w2_input_scale = torch.ones_like(quant_config.g2_alphas)
            experts.process_weights_after_loading(fake_layer)
        else:
            experts = FlashInferExperts(moe_config, quant_config)

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        )
        actual = kernel.apply(
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        ).clone()

        w1 = _dequantize_weights(
            w1_q, quant_config.w1_scale, w1_global_encode_scale, dtype
        )
        w2 = _dequantize_weights(
            w2_q, quant_config.w2_scale, w2_global_encode_scale, dtype
        )
        expected = _reference(
            hidden_states,
            w1,
            w2,
            topk_ids,
            quant_config.a1_gscale,
            a2_gscale,
        )
        pooled = _reference(
            hidden_states,
            w1,
            w2,
            topk_ids,
            quant_config.a1_gscale,
            torch.full_like(a2_gscale, a2_gscale.min()),
        )

        # Repeat with the historical pooled scale. Expert 0 is unchanged;
        # outputs for experts whose scale differs must change. This direct
        # kernel-to-kernel comparison is insensitive to GEMM accumulation
        # differences between the CUDA kernel and the PyTorch reference.
        quant_config.a2_gscale.fill_(a2_gscale[0])
        quant_config.g2_alphas.copy_(1.0 / w2_global_encode_scale)
        if backend == "flashinfer_trtllm":
            assert isinstance(experts, TrtLlmNvFp4ExpertsModular)
            experts.g1_scale_c.copy_(quant_config.g1_alphas * quant_config.a2_gscale)
        actual_pooled = kernel.apply(
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        ).clone()

        actual_changed = (actual != actual_pooled).any(dim=1)
        expected_changed = (expected != pooled).any(dim=1)
        assert not actual_changed[0] and not expected_changed[0]
        assert torch.equal(actual_changed, expected_changed), (
            f"{backend} changed rows {actual_changed.tolist()} do not match "
            f"the per-expert reference {expected_changed.tolist()}"
        )
        torch.testing.assert_close(actual, expected, atol=3e-1, rtol=2e-1)
        torch.testing.assert_close(actual_pooled, pooled, atol=3e-1, rtol=2e-1)
