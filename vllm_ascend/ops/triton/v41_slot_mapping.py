# SPDX-License-Identifier: Apache-2.0
"""Fused equivalent of the V4.1 non-packed slot preparation path."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _slots_kernel(S, P, Q, C, F, N, R, T, SS, PS, QS, SKIP: tl.constexpr,
                  HAS_POS: tl.constexpr, RATIO: tl.constexpr,
                  PAGE: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = i < N
    s = tl.load(S + i * SS, mask, other=-1).to(tl.int64)
    valid = s >= 0
    if RATIO != 1:
        valid &= ((s + 1) % RATIO) == 0
        s = s // RATIO
    if RATIO == 2:
        if SKIP:
            valid &= False
        else:
            end = tl.minimum(tl.load(Q + R * QS), T)
            valid &= i < end
            if HAS_POS:
                pos = tl.load(P + i * PS, mask, other=0)
                valid &= (pos % 2) == 1
    physical = tl.maximum(s, 0)
    tl.store(C + 2 * i, tl.where(valid, physical // PAGE, -1), mask)
    tl.store(C + 2 * i + 1, tl.where(valid, physical % PAGE, -1), mask)
    tl.store(F + i, tl.where(valid, s, -1), mask)


def build_v41_slot_mapping(slots, positions, query_start_loc, num_tokens,
                           num_actual_reqs, num_actual_tokens, page_size,
                           ratio, skip_update, coordinates_output, flat_output):
    # Outputs are contiguous persistent views owned by the metadata builder.
    if num_tokens:
        _slots_kernel[(triton.cdiv(num_tokens, 256),)](
            slots, positions if positions is not None else slots,
            query_start_loc, coordinates_output, flat_output,
            num_tokens, num_actual_reqs, num_actual_tokens, slots.stride(0),
            positions.stride(0) if positions is not None else slots.stride(0),
            query_start_loc.stride(0), SKIP=bool(skip_update),
            HAS_POS=positions is not None, RATIO=ratio, PAGE=page_size, TILE=256,
        )
    return coordinates_output, flat_output
