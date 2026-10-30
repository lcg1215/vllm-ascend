# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""8-card HCCL helper smoke test, not a real-model or ACL graph benchmark.

torchrun --standalone --nproc-per-node=8 tools/check_v41_local_argmax_hccl.py
"""

import os
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.spec_decode.dspark.local_argmax_v41 import reduce_local_argmax

rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(rank)
dist.init_process_group("hccl")
if dist.get_world_size() != 8:
    raise ValueError("Run this check with exactly eight ranks")
groups = [dist.new_group(list(range(0, 4))), dist.new_group(list(range(4, 8)))]
group = groups[dist.get_rank() // 4]
tp_rank = dist.get_rank() % 4
batch, steps, vocab = 16, 5, 129281
# Same inputs for the two replica groups. CPU-generated tensors avoid relying
# on any cross-device RNG implementation for this smoke test.
gen = torch.Generator().manual_seed(91)
base = torch.randn(batch, steps, vocab, generator=gen, dtype=torch.bfloat16).npu()
w1 = torch.randn(vocab, 32, generator=gen, dtype=torch.bfloat16).npu()
w2 = torch.randn(vocab, 32, generator=gen, dtype=torch.bfloat16).npu()
prev = torch.arange(batch, device="npu", dtype=torch.int64)
reference_prev = prev.clone()
width = (vocab + 3) // 4
start, end = min(tp_rank * width, vocab), min((tp_rank + 1) * width, vocab)
for step in range(steps):
    full_bias = w1[prev] @ w2.t()
    reference = (base[:, step] + w1[reference_prev] @ w2.t()).argmax(-1)

    def gather(pair):
        buffers = [torch.empty_like(pair) for _ in range(4)]
        dist.all_gather(buffers, pair.contiguous(), group=group)
        return torch.cat(buffers, -1)

    prev = reduce_local_argmax(
        base[:, step, start:end] + full_bias[:, start:end],
        vocab_start=start,
        vocab_end=end,
        vocab_size=vocab,
        tp_size=4,
        all_gather=gather,
    )
    torch.testing.assert_close(prev, reference, rtol=0, atol=0)
    reference_prev = reference
results = [torch.empty_like(prev) for _ in range(8)]
dist.all_gather(results, prev)
for result in results:
    torch.testing.assert_close(result, prev, rtol=0, atol=0)
torch.npu.synchronize()
if dist.get_rank() == 0:
    print("PASS: TP4 candidate reduction, two replica groups, five Markov steps")
dist.destroy_process_group()
