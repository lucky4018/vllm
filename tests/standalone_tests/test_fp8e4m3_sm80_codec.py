# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contracts for the SM80 OCP FP8 E4M3FN codec."""

import importlib.util
import math
import random
import struct
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

REPO_ROOT = Path(__file__).parents[2]
CODEC_PATH = REPO_ROOT / "vllm" / "_fp8e4m3_sm80.py"
MQA_LOGITS_PATH = (
    REPO_ROOT / "vllm" / "v1" / "attention" / "ops" / "mqa_logits_triton.py"
)


class _Scalar:
    """Tiny scalar interpreter for the codec's elementwise Triton expression."""

    def __init__(self, value):
        self.value = value.value if isinstance(value, _Scalar) else value

    def to(self, dtype, bitcast=False):
        if bitcast:
            if dtype == "int32":
                value = struct.unpack("<i", struct.pack("<f", float(self.value)))[0]
            elif dtype == "float32":
                value = struct.unpack(
                    "<f", struct.pack("<I", int(self.value) & 0xFFFFFFFF)
                )[0]
            else:
                raise AssertionError(f"unsupported bitcast: {dtype}")
        elif dtype == "float32":
            value = struct.unpack("<f", struct.pack("<f", float(self.value)))[0]
        elif dtype == "uint8":
            value = int(self.value) & 0xFF
        else:
            value = int(self.value)
        return _Scalar(value)

    def _binary(self, other, operation):
        other = other.value if isinstance(other, _Scalar) else other
        return _Scalar(operation(self.value, other))

    def __add__(self, other):
        return self._binary(other, lambda left, right: left + right)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, lambda left, right: left - right)

    def __rsub__(self, other):
        return _Scalar(other)._binary(self, lambda left, right: left - right)

    def __mul__(self, other):
        return self._binary(other, lambda left, right: left * right)

    __rmul__ = __mul__

    def __lshift__(self, other):
        return self._binary(other, lambda left, right: left << right)

    def __rlshift__(self, other):
        return _Scalar(other).__lshift__(self)

    def __rshift__(self, other):
        return self._binary(other, lambda left, right: left >> right)

    def __and__(self, other):
        return self._binary(other, lambda left, right: left & right)

    __rand__ = __and__

    def __or__(self, other):
        return self._binary(other, lambda left, right: left | right)

    __ror__ = __or__

    def __lt__(self, other):
        return self._binary(other, lambda left, right: left < right)

    def __le__(self, other):
        return self._binary(other, lambda left, right: left <= right)

    def __gt__(self, other):
        return self._binary(other, lambda left, right: left > right)

    def __ge__(self, other):
        return self._binary(other, lambda left, right: left >= right)

    def __eq__(self, other):
        return self._binary(other, lambda left, right: left == right)

    def __ne__(self, other):
        return self._binary(other, lambda left, right: left != right)

    def __bool__(self):
        return bool(self.value)


def _unwrap(value):
    return value.value if isinstance(value, _Scalar) else value


def _load_codec() -> ModuleType:
    """Load the codec without importing vLLM's optional runtime dependencies."""
    triton = ModuleType("vllm.triton_utils.triton")
    triton.jit = lambda fn: fn
    triton_utils = ModuleType("vllm.triton_utils")
    tl = ModuleType("vllm.triton_utils.tl")
    tl.int32 = "int32"
    tl.float32 = "float32"
    tl.uint8 = "uint8"
    tl.abs = lambda value: _Scalar(abs(_unwrap(value)))
    tl.exp2 = lambda value: _Scalar(2 ** _unwrap(value))
    tl.maximum = lambda left, right: _Scalar(max(_unwrap(left), _unwrap(right)))
    tl.minimum = lambda left, right: _Scalar(min(_unwrap(left), _unwrap(right)))
    tl.where = lambda condition, left, right: _Scalar(
        _unwrap(left) if _unwrap(condition) else _unwrap(right)
    )
    triton_utils.tl = tl
    triton_utils.triton = triton
    vllm = ModuleType("vllm")
    vllm.__path__ = [str(REPO_ROOT / "vllm")]

    name = "vllm._fp8e4m3_sm80_test_module"
    spec = importlib.util.spec_from_file_location(name, CODEC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "vllm": vllm,
            "vllm.triton_utils": triton_utils,
        },
    ):
        spec.loader.exec_module(module)
    return module


CODEC = _load_codec()


def _expected_decode(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return math.nan
    if exponent == 0:
        return sign * mantissa * 2**-9
    return sign * (8 + mantissa) * 2 ** (exponent - 10)


def _finite_codes() -> list[tuple[float, int]]:
    return [(_expected_decode(byte), byte) for byte in range(0x7F)]


def _expected_encode(value: float) -> int:
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    if math.isnan(value):
        return sign | 0x7F
    if math.isinf(value):
        return sign | 0x7E

    absolute = min(abs(value), 448.0)
    candidates = _finite_codes()
    best_distance = min(abs(absolute - decoded) for decoded, _ in candidates)
    tied = [
        byte
        for decoded, byte in candidates
        if abs(abs(absolute - decoded) - best_distance) < 1e-15
    ]
    # Round-to-nearest-even: the retained E4M3 mantissa's low bit is even.
    byte = next(
        (candidate for candidate in tied if (candidate & 0x7) % 2 == 0),
        tied[0],
    )
    return sign | byte


def test_decode_exhaustively_matches_ocp_e4m3fn_definition():
    for byte in range(256):
        actual = CODEC._e4m3fn_to_f32_reference(byte)
        expected = _expected_decode(byte)
        if math.isnan(expected):
            assert math.isnan(actual), hex(byte)
        else:
            assert actual == expected, (hex(byte), actual, expected)


def test_triton_decode_expression_matches_reference_for_all_bytes():
    for byte in range(256):
        actual = _unwrap(CODEC.e4m3fn_to_f32(_Scalar(byte)))
        expected = CODEC._e4m3fn_to_f32_reference(byte)
        if math.isnan(expected):
            assert math.isnan(actual), hex(byte)
            assert math.copysign(1.0, actual) == math.copysign(1.0, expected)
        else:
            assert actual == expected, (hex(byte), actual, expected)


def test_every_finite_byte_round_trips_and_nan_bytes_stay_nan():
    for byte in range(256):
        decoded = CODEC._e4m3fn_to_f32_reference(byte)
        encoded = CODEC._f32_to_e4m3fn_reference(decoded)
        if byte in (0x7F, 0xFF):
            assert encoded == byte
        else:
            assert encoded == byte, hex(byte)


def test_encode_preserves_nan_sign_zero_and_saturates_nonfinite_values():
    values_and_expected = (
        (0.0, 0x00),
        (-0.0, 0x80),
        (math.nan, 0x7F),
        (-math.nan, 0xFF),
        (math.inf, 0x7E),
        (-math.inf, 0xFE),
        (500.0, 0x7E),
        (-500.0, 0xFE),
    )
    for value, expected in values_and_expected:
        assert CODEC._f32_to_e4m3fn_reference(value) == expected


def test_encode_uses_round_to_nearest_even_at_every_finite_boundary():
    for lower_byte in range(0x7E):
        upper_byte = lower_byte + 1
        midpoint = (
            _expected_decode(lower_byte) + _expected_decode(upper_byte)
        ) / 2
        expected = lower_byte if lower_byte & 1 == 0 else upper_byte
        assert CODEC._f32_to_e4m3fn_reference(midpoint) == expected
        assert CODEC._f32_to_e4m3fn_reference(-midpoint) == expected | 0x80


def test_triton_encode_expression_matches_reference_at_every_boundary():
    values = [0.0, -0.0, math.nan, -math.nan, math.inf, -math.inf]
    for lower_byte in range(0x7E):
        midpoint = (
            _expected_decode(lower_byte) + _expected_decode(lower_byte + 1)
        ) / 2
        values.extend((midpoint, -midpoint))
    for value in values:
        actual = _unwrap(CODEC.f32_to_e4m3fn(_Scalar(value).to("float32")))
        assert actual == CODEC._f32_to_e4m3fn_reference(value), value


def test_encode_matches_nearest_even_reference_for_deterministic_float_values():
    rng = random.Random(0xE4A3)
    values = [0.0, -0.0, 2**-30, 2**-10, 2**-9, 448.0, -448.0]
    values.extend((rng.random() - 0.5) * 2000 for _ in range(512))
    for value in values:
        assert CODEC._f32_to_e4m3fn_reference(value) == _expected_encode(value)


def test_mqa_logits_lut_does_not_reinterpret_nan_bytes_as_finite_values():
    source = MQA_LOGITS_PATH.read_text()
    assert "lut[0x7F]" not in source
    assert "lut[0xFF]" not in source
