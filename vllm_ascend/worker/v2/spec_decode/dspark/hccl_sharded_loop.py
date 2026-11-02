# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental HCCL entry point for upstream vLLM #63886.

Requires the #64356 NPU cache/read/write adapter first.
No changes to upstream ownership, stale-draft handling, or Markov sampling.
"""
import os

import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.logger import init_logger

from vllm_ascend.utils import lmhead_tp_enable

logger = init_logger(__name__)


class _HcclDraftGather:
    def all_gather(self, output, input):
        # Same HCCL ProcessGroup as upstream all_to_all_vocab_shards fallback.
        # This is an eager-first implementation; NPU graph capture is unverified.
        dist.all_gather_into_tensor(output, input, group=get_tp_group().device_group)


class HcclDSparkShardedLoopMixin:
    def _try_enable_sharded_sequential_sampling(self) -> None:
        if os.getenv("VLLM_ASCEND_DSPARK_SHARDED_LOOP", "0") != "1":
            return super()._try_enable_sharded_sequential_sampling()
        group = get_tp_group()
        pc = self.vllm_config.parallel_config
        if any(getattr(pc, name, 1) != 1 for name in (
            "data_parallel_size", "pipeline_parallel_size",
            "prefill_context_parallel_size", "decode_context_parallel_size",
        )):
            raise ValueError("Initial HCCL port requires DP=PP=PCP=DCP=1.")
        if lmhead_tp_enable():
            raise ValueError("Fine-grained LM-head TP is outside this port's scope.")
        if type(self.model).__name__ != "DSparkDeepseekV41ForCausalLM":
            raise ValueError("Initial HCCL port is scoped to Ascend DeepSeek V4.1.")
        if self.draft_logits is not None and self.draft_logits.row_mapping is None:
            raise ValueError("Probabilistic drafting requires a sharded DraftLogitsCache.")
        if self.use_acceptance_estimator or self.use_confidence_head:
            raise ValueError("Initial HCCL port requires adaptive verification OFF.")
        if self._draft_topk is not None or hasattr(self.model, "sample_sequential"):
            raise ValueError("Top-k and model-fused Markov paths are unsupported.")
        if self.draft_watermarker is not None:
            raise ValueError("Draft watermarking is unsupported.")
        if (
            not hasattr(self.model, "compute_draft_logits_local")
            or self.model.lm_head.tp_size != group.world_size
            or self.model.model.markov_head.markov_w2.tp_size != 1
        ):
            raise ValueError("Requires ordinary TP draft head and replicated Markov head.")
        if not self.speculative_config.enforce_eager and os.getenv(
            "VLLM_ASCEND_DSPARK_SHARDED_LOOP_ALLOW_GRAPH", "0"
        ) != "1":
            raise ValueError(
                "Validate draft eager first. NPU graph capture is unverified; "
                "experimental opt-in: VLLM_ASCEND_DSPARK_SHARDED_LOOP_ALLOW_GRAPH=1."
            )
        # Upstream _sample_sequential_sharded only uses this object's all_gather.
        # Its logits all-to-all independently uses the c10d/HCCL fallback.
        self._sharded_sequential_comm = _HcclDraftGather()
        logger.info(
            "Ascend DSpark upstream batch-sharded Markov loop enabled with HCCL, TP=%d.",
            group.world_size,
        )
