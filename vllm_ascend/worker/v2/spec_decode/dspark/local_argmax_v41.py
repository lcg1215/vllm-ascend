# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 V2 DSpark local-vocabulary greedy reduction.

Adapted from the idea in vllm-ascend PR #18824 by lcg1215.
Keep the existing full-vocabulary Markov GEMM; only the base LM-head
all-gather is replaced by one candidate-pair gather per draft step.
"""

from collections.abc import Callable

import torch
from vllm.distributed import get_tp_group
from vllm.logger import init_logger

from vllm_ascend.models.deepseek_v41.dspark import DSparkDeepseekV41ForCausalLM
from vllm_ascend.utils import lmhead_tp_enable

logger = init_logger(__name__)
MAX_EXACT_FP32_TOKEN_ID = 1 << 24


def reduce_local_argmax(
    scores: torch.Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    vocab_size: int,
    tp_size: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return full-vocab greedy IDs, including ties and empty padded shards.

    Token IDs and scores travel in one FP32 pair; the sentinel must also be
    exactly representable. Inputs are not modified. No device-to-host reads.
    """
    if not 0 < vocab_size <= MAX_EXACT_FP32_TOKEN_ID:
        raise ValueError("Vocabulary exceeds the exact FP32 token-ID range")
    valid = vocab_end - vocab_start
    if not 0 <= vocab_start <= vocab_end <= vocab_size or valid > scores.shape[-1]:
        raise ValueError("Invalid local vocabulary shard")
    rows = scores.shape[0]
    if rows == 0:
        return torch.empty(0, dtype=torch.int64, device=scores.device)
    if valid:
        values, ids = scores[:, :valid].max(dim=-1)
        ids = ids + vocab_start
        values = values.float()
    else:
        values = torch.full((rows,), float("-inf"), device=scores.device, dtype=torch.float32)
        ids = torch.full((rows,), MAX_EXACT_FP32_TOKEN_ID, device=scores.device, dtype=torch.int64)
    pair = torch.stack((values, ids.float()), dim=-1)
    gathered = pair if tp_size == 1 else all_gather(pair)
    candidates = gathered.view(rows, tp_size, 2)
    candidate_scores = candidates[..., 0]
    candidate_ids = candidates[..., 1]
    nan_mask = torch.isnan(candidate_scores)
    has_nan = nan_mask.any(dim=-1, keepdim=True)
    maximum = torch.where(nan_mask, float("-inf"), candidate_scores).amax(dim=-1, keepdim=True)
    winners = torch.where(has_nan, nan_mask, candidate_scores == maximum)
    # Explicit token-ID tie breaking is independent of rank order.
    ids = torch.where(winners, candidate_ids, float(MAX_EXACT_FP32_TOKEN_ID))
    return ids.amin(dim=-1).to(torch.int64)


class V41LocalArgmaxMixin:
    """Scoped V4.1 support; other models keep upstream validation/sampling."""

    @staticmethod
    def _unwrap_v41_model(model):
        seen = set()
        while hasattr(model, "get_original_model"):
            if id(model) in seen:
                raise ValueError("Cycle in DSpark model wrappers")
            seen.add(id(model))
            original = model.get_original_model()
            if original is model:
                break
            model = original
        return model

    def _validate_local_argmax_reduction(self):
        if not getattr(self, "use_local_argmax_reduction", False):
            return super()._validate_local_argmax_reduction()
        model = self._unwrap_v41_model(self.model)
        if not isinstance(model, DSparkDeepseekV41ForCausalLM):
            return super()._validate_local_argmax_reduction()
        if self.speculative_config.draft_sample_method != "greedy":
            raise ValueError("V4.1 local argmax requires draft_sample_method='greedy'")
        if getattr(self, "_draft_topk", None) is not None:
            raise ValueError("V4.1 local argmax does not support dspark_draft_topk")
        if model.draft_id_to_target_id is not None:
            raise ValueError("V4.1 local argmax requires identity vocabulary mapping")
        if lmhead_tp_enable():
            raise ValueError("V4.1 local argmax does not support fine-grained LM-head TP")
        if getattr(getattr(self, "vllm_config", None), "lora_config", None) is not None:
            raise ValueError("V4.1 local argmax does not support LoRA vocabulary layouts")
        if getattr(self, "use_acceptance_estimator", False):
            raise ValueError("V4.1 local argmax requires a confidence head for adaptive verification")
        if getattr(self, "draft_watermarker", None) is not None or hasattr(model, "apply_output_logit_transform"):
            raise ValueError("V4.1 local argmax does not support draft output transforms")
        if not hasattr(model, "compute_local_draft_logits"):
            raise ValueError("V4.1 draft has no compute_local_draft_logits method")
        head, group = model.lm_head, get_tp_group()
        if head.tp_size != group.world_size or head.tp_rank != group.rank_in_group:
            raise ValueError("LM-head TP layout must match the model TP group")
        head_group = getattr(head, "comm_group", None)
        if head_group is not None and head_group.unique_name != group.unique_name:
            raise ValueError("LM-head communication group must match the model TP group")
        if head.org_vocab_size != model.config.vocab_size or head.num_added_embeddings != 0:
            raise ValueError("Only the original contiguous vocabulary layout is supported")
        if not 0 < model.config.vocab_size <= MAX_EXACT_FP32_TOKEN_ID:
            raise ValueError("Vocabulary exceeds the exact FP32 token-ID range")
        logger.info(
            "V4.1 DSpark local argmax enabled: TP=%d; greedy pair reduction, full Markov GEMM retained.",
            group.world_size,
        )

    def _sample_sequential(self, num_reqs, head_hidden):
        if not getattr(self, "use_local_argmax_reduction", False):
            return super()._sample_sequential(num_reqs, head_hidden)
        model = self._unwrap_v41_model(self.model)
        if not isinstance(model, DSparkDeepseekV41ForCausalLM):
            return super()._sample_sequential(num_reqs, head_hidden)
        if num_reqs == 0:
            return
        runtime_model = self.model
        n_spec = self.num_speculative_steps
        hidden = head_hidden[self.sample_indices[: num_reqs * n_spec]]
        # Call through the runtime wrapper; ModelWithContext.__getattr__
        # forwards this method in the pinned main branch.
        local = runtime_model.compute_local_draft_logits(hidden)
        local = local.view(num_reqs, n_spec, local.shape[-1])
        shard = model.lm_head.shard_indices
        start, end = shard.org_vocab_start_index, shard.org_vocab_end_index
        group = get_tp_group()
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        embeds = []
        for step in range(n_spec):
            embed = runtime_model.markov_embed(prev)
            if self.use_confidence_head:
                embeds.append(embed)
            bias = runtime_model.markov_bias(embed)
            scores = local[:, step, : end - start] + bias[:, start:end]
            sampled = reduce_local_argmax(
                scores,
                vocab_start=start,
                vocab_end=end,
                vocab_size=model.config.vocab_size,
                tp_size=group.world_size,
                all_gather=lambda pair: group.all_gather(pair, dim=-1),
            )
            self.draft_tokens[:num_reqs, step].copy_(sampled)
            prev = sampled
        if self.use_confidence_head:
            confidence = runtime_model.compute_confidence(hidden, torch.stack(embeds, dim=1).flatten(0, 1))
            self.draft_token_confidence_probs[:num_reqs].copy_(confidence.view(num_reqs, n_spec))
