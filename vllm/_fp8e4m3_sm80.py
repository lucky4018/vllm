# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SM80 (Ampere / A800, cap 8.0) software codec for OCP FP8 E4M3FN.

Triton on SM80 does not support the ``tl.float8e4nv`` (e4m3fn) dtype
(only ``fp8e4b15`` / ``fp8e5``). DeepSeek-V4 stores its paged K-cache and
compressed-KV cache as *real* e4m3fn bytes (written by several kernels,
read back by one dequant kernel), so we cannot switch to a private 8-bit
format. Instead we emulate e4m3fn exactly in fp32 integer math:

  * ``e4m3fn_to_f32(u)``  : raw uint8 e4m3fn byte  -> fp32 value
  * ``f32_to_e4m3fn(x)``  : fp32 value             -> raw uint8 e4m3fn byte

E4M3FN: 1 sign / 4 exp (bias 7) / 3 mantissa, no inf, max finite = 448,
S.1111.111 = NaN. Encoder uses round-to-nearest-even and clamps to +-448.
Finite values are byte-exact inverses, so kernels using the native dtype on
SM90 and this codec on SM80 round-trip identically.
"""

import math
import struct

from vllm.triton_utils import tl, triton


def _e4m3fn_to_f32_reference(byte: int) -> float:
    """CPU reference decoder used to validate the Triton expression."""
    byte &= 0xFF
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return math.copysign(math.nan, sign)
    if exponent == 0:
        return sign * mantissa * 2**-9
    return sign * (8 + mantissa) * 2 ** (exponent - 10)


def _f32_to_e4m3fn_reference(value: float) -> int:
    """CPU reference encoder matching PyTorch's saturating E4M3FN cast."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = bits >> 24 & 0x80
    magnitude = bits & 0x7FFFFFFF
    exponent = (magnitude >> 23) & 0xFF
    mantissa = magnitude & 0x7FFFFF

    if exponent == 0xFF:
        return sign | (0x7F if mantissa else 0x7E)

    # The E4M3 subnormal quantum is 2^-9. This fixed-point form also handles
    # the transition to the smallest normal without a separate boundary case.
    unbiased_exponent = exponent - 127
    if unbiased_exponent < -6:
        if unbiased_exponent < -10 or exponent == 0:
            rounded = 0
        else:
            significand = mantissa | 0x800000
            shift = 14 - unbiased_exponent
            retained = significand >> shift
            remainder = significand & ((1 << shift) - 1)
            halfway = 1 << (shift - 1)
            rounded = retained + (
                remainder > halfway
                or (remainder == halfway and retained & 1)
            )
        return sign | rounded

    retained = mantissa >> 20
    remainder = mantissa & 0xFFFFF
    halfway = 1 << 19
    retained += remainder > halfway or (
        remainder == halfway and retained & 1
    )
    e4m3_exponent = unbiased_exponent + 7 + (retained >> 3)
    e4m3_mantissa = retained & 0x7
    if e4m3_exponent > 0xF or (
        e4m3_exponent == 0xF and e4m3_mantissa == 0x7
    ):
        return sign | 0x7E
    return sign | e4m3_exponent << 3 | e4m3_mantissa


@triton.jit
def e4m3fn_to_f32(u):
    """Decode a raw e4m3fn byte (held in an integer tensor) to fp32."""
    u = u.to(tl.int32)
    s = (u >> 7) & 1
    e = (u >> 3) & 0xF
    m = u & 0x7
    sign = (1 - 2 * s).to(tl.float32)
    # subnormal (e==0): sign * m * 2^-9
    sub = sign * m.to(tl.float32) * 0.001953125  # 2^-9
    # normal       : sign * (8+m) * 2^(e-10)
    nrm = sign * (8 + m).to(tl.float32) * tl.exp2((e - 10).to(tl.float32))
    decoded = tl.where(e == 0, sub, nrm)
    is_nan = (e == 0xF) & (m == 0x7)
    signed_nan_bits = 0x7FC00000 | (s << 31)
    nan = signed_nan_bits.to(tl.float32, bitcast=True)
    return tl.where(is_nan, nan, decoded)


@triton.jit
def f32_to_e4m3fn(x):
    """Encode fp32 to a raw e4m3fn byte (uint8) with round-to-nearest-even."""
    x_bits = x.to(tl.int32, bitcast=True)
    s = (x_bits >> 31) & 1
    is_nan = ((x_bits >> 23) & 0xFF) == 0xFF
    is_nan = is_nan & ((x_bits & 0x7FFFFF) != 0)
    ax = tl.minimum(tl.abs(x), 448.0)
    # Keep integer reinterpretation defined for NaN lanes; the final byte is
    # selected from ``is_nan`` below, so this value is only a safe placeholder.
    ax = tl.where(is_nan, 0.0, ax)

    bits = ax.to(tl.int32, bitcast=True)          # ax >= 0
    E32 = ((bits >> 23) & 0xFF) - 127             # unbiased fp32 exponent
    M = bits & 0x7FFFFF                           # 23-bit fraction

    # ---- normal target: e_field = E32 + 7 in [1,15] ----
    efield = E32 + 7
    m_top = M >> 20                               # top 3 fraction bits (0..7)
    round_bit = (M >> 19) & 1
    sticky = (M & 0x7FFFF) != 0
    roundup = round_bit & (sticky | (m_top & 1))  # round-half-to-even
    m_n = m_top + roundup                         # 0..8
    efield_n = efield + (m_n >> 3)                # mantissa carry -> exp+1
    m_n = m_n & 0x7

    # ---- subnormal target (efield <= 0, ax < 2^-6) ----
    # value = m_sub * 2^-9, m_sub = round(ax * 512) via fixed-point shift.
    # Clamp E32 so the shift count / round constant never overflow int32;
    # results for clamped (tiny) inputs are masked to zero below.
    E32c = tl.minimum(tl.maximum(E32, -10), -7)
    full = M | 0x800000                           # 1.fraction as 24-bit
    rsh = 14 - E32c                               # in [21, 24]
    retained_sub = full >> rsh
    remainder_sub = full & ((1 << rsh) - 1)
    halfway_sub = 1 << (rsh - 1)
    roundup_sub = (remainder_sub > halfway_sub) | (
        (remainder_sub == halfway_sub) & ((retained_sub & 1) != 0)
    )
    m_sub = retained_sub + roundup_sub             # round-to-nearest-even, 0..8
    m_sub = tl.minimum(m_sub, 8)

    is_sub = efield <= 0
    efield_f = tl.where(is_sub, tl.where(m_sub >= 8, 1, 0), efield_n)
    m_f = tl.where(is_sub, tl.where(m_sub >= 8, 0, m_sub), m_n)

    # Below 2^-10, and at the even tie itself, round to signed zero.
    tiny = ax <= 0.0009765625
    efield_f = tl.where(tiny, 0, efield_f)
    m_f = tl.where(tiny, 0, m_f)

    # 0x7f/0xff are NaN, not +/-480. Saturate finite overflow to +/-448.
    finite_overflow = (efield_f > 15) | ((efield_f == 15) & (m_f >= 7))
    efield_f = tl.where(finite_overflow, 15, efield_f)
    m_f = tl.where(finite_overflow, 6, m_f)
    efield_f = tl.maximum(tl.minimum(efield_f, 15), 0)
    m_f = tl.maximum(tl.minimum(m_f, 7), 0)
    byte = (s << 7) | (efield_f << 3) | m_f
    byte = tl.where(is_nan, (s << 7) | 0x7F, byte)
    return byte.to(tl.uint8)
