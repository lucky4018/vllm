# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear import (
    init_wfp8_a16_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    STRATEGY_TO_PARAMETER_TYPE,
    STRATEGY_TO_WEIGHT_QUANT_KEY,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import replace_parameter

__all__ = ["CompressedTensorsW8A16Fp8"]


class CompressedTensorsW8A16Fp8(CompressedTensorsScheme):
    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure

        self.weight_quant_key = STRATEGY_TO_WEIGHT_QUANT_KEY[self.strategy]
        self.activation_quant_key = (
            kFp8StaticTensorSym if is_static_input_scheme else kFp8DynamicTensorSym
        )

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            # Validate block quantization shapes
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = create_fp8_scale_parameter(
            STRATEGY_TO_PARAMETER_TYPE[self.strategy],
            output_partition_sizes,
            input_size_per_partition,
            layer.weight_block_size,
            weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (to deal with converted checkpoints)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

        self.linear_kernel = init_wfp8_a16_linear_kernel(
            weight_quant_key=self.weight_quant_key,
            activation_quant_key=self.activation_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
        )

    def _sm80_dequant_w8a16(self) -> bool:
        from vllm.platforms import current_platform
        if not current_platform.is_cuda():
            return False
        cap = current_platform.get_device_capability()
        return (cap is not None and cap.major < 9
                and self.strategy != QuantizationStrategy.BLOCK)

    def _do_sm80_dequant(self, layer: torch.nn.Module) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            replace_parameter(
                layer, "weight_scale",
                convert_to_channelwise(layer.weight_scale, layer.logical_widths))
        ws = layer.weight_scale.data.reshape(-1, 1).to(torch.float32)
        w = layer.weight.data.to(torch.float32) * ws
        replace_parameter(layer, "weight", w.to(torch.bfloat16))
        for p in ("weight_scale", "input_scale"):
            if p in layer._parameters and layer._parameters[p] is not None:
                del layer._parameters[p]
        layer._sm80_bf16 = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._sm80_dequant_w8a16():
            self._do_sm80_dequant(layer)
            return
        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            # MarlinFP8ScaledMMLinearKernel uses "weight_scale_inv" for block
            # quant, while CT registers the scale as "weight_scale".
            # Rename by deleting the old parameter and adding the new one so
            # that prepare_fp8_layer_for_marlin (which prefers "weight_scale"
            # over "weight_scale_inv") picks up "weight_scale_inv" correctly.
            weight_scale_data = layer.weight_scale.data
            del layer._parameters["weight_scale"]
            replace_parameter(layer, "weight_scale_inv", weight_scale_data)
        else:
            if self.strategy == QuantizationStrategy.TENSOR:
                # For fused modules with per-tensor scales, expand each scale
                # to its shard's channels.
                replace_parameter(
                    layer,
                    "weight_scale",
                    convert_to_channelwise(layer.weight_scale, layer.logical_widths),
                )
            # Canonicalize to (K, N) for the kernel.
            replace_parameter(layer, "weight", layer.weight.t())

        self.linear_kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_sm80_bf16", False):
            import torch.nn.functional as F
            return F.linear(x.to(layer.weight.dtype), layer.weight, bias)
        return self.linear_kernel.apply_weights(layer, x, bias)
