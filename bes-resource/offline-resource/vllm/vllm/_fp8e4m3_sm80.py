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
These are byte-exact inverses on every representable value, so any kernel
mix (native e4m3fn on SM90 vs this codec on SM80) round-trips identically.
"""

from vllm.triton_utils import tl, triton


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
    return tl.where(e == 0, sub, nrm)


@triton.jit
def f32_to_e4m3fn(x):
    """Encode fp32 to a raw e4m3fn byte (uint8) with round-to-nearest-even."""
    s = tl.where(x < 0, 1, 0).to(tl.int32)
    ax = tl.minimum(tl.abs(x), 448.0)

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
    E32c = tl.maximum(E32, -10)
    full = M | 0x800000                           # 1.fraction as 24-bit
    rsh = 14 - E32c                               # in [21, 24]
    m_sub = (full + (1 << (rsh - 1))) >> rsh      # round-half-up, 0..8
    m_sub = tl.minimum(m_sub, 8)

    is_sub = efield <= 0
    efield_f = tl.where(is_sub, tl.where(m_sub >= 8, 1, 0), efield_n)
    m_f = tl.where(is_sub, tl.where(m_sub >= 8, 0, m_sub), m_n)

    # round-to-zero region: |x| <= 2^-10 (the tie at 2^-10 rounds to even = 0)
    tiny = ax <= 0.0009765625
    efield_f = tl.where(tiny, 0, efield_f)
    m_f = tl.where(tiny, 0, m_f)

    efield_f = tl.maximum(tl.minimum(efield_f, 15), 0)
    m_f = tl.maximum(tl.minimum(m_f, 7), 0)
    byte = (s << 7) | (efield_f << 3) | m_f
    return byte.to(tl.uint8)
