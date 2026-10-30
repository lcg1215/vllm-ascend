# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the actual helper/mixin without model construction."""

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")


def _implementation():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dspark/local_argmax_v41.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    scope = {
        "torch": torch,
        "Callable": Callable,
        "MAX_EXACT_FP32_TOKEN_ID": 1 << 24,
        "DSparkDeepseekV41ForCausalLM": type("Draft", (), {}),
        "lmhead_tp_enable": lambda: False,
        "logger": Mock(),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_reduction_matches_dense(dtype):
    scope = _implementation()
    reduce = scope["reduce_local_argmax"]
    torch.manual_seed(4)
    scores = torch.randn(16, 19, dtype=dtype)
    scores[0, 1] = scores[0, 12] = 100
    scores[1].fill_(float("-inf"))
    scores[2, 8] = scores[2, 15] = float("nan")
    boundaries = [0, 5, 10, 15, 19]
    pairs = []
    for start, end in zip(boundaries, boundaries[1:]):
        vals, ids = scores[:, start:end].max(-1)
        pairs.append(torch.stack([vals.float(), (ids + start).float()], -1))
    # Deliberately permute rank order; ties still choose smallest token ID.
    gathered = torch.cat([pairs[i] for i in [3, 1, 0, 2]], -1)
    for start, end in zip(boundaries, boundaries[1:]):
        local = torch.cat([scores[:, start:end], torch.full((16, 2), 1000, dtype=dtype)], -1)
        before = local.clone()
        out = reduce(
            local, vocab_start=start, vocab_end=end, vocab_size=19, tp_size=4, all_gather=lambda pair: gathered
        )
        torch.testing.assert_close(out, scores.argmax(-1), rtol=0, atol=0)
        torch.testing.assert_close(local, before, equal_nan=True)


def test_empty_vocab_shard_and_zero_rows():
    reduce = _implementation()["reduce_local_argmax"]
    remote = torch.tensor([[float("-inf"), 0.0]])
    out = reduce(
        torch.empty(1, 2),
        vocab_start=3,
        vocab_end=3,
        vocab_size=3,
        tp_size=2,
        all_gather=lambda pair: torch.cat([pair, remote], -1),
    )
    assert out.tolist() == [0]
    gather = Mock()
    out = reduce(torch.empty(0, 3), vocab_start=0, vocab_end=3, vocab_size=3, tp_size=1, all_gather=gather)
    assert out.shape == (0,)
    gather.assert_not_called()
    with pytest.raises(ValueError, match="FP32"):
        reduce(torch.zeros(1, 2), vocab_start=0, vocab_end=2, vocab_size=(1 << 24) + 1, tp_size=1, all_gather=gather)


def test_five_step_markov_confidence_and_wrapper():
    scope = _implementation()
    batch, steps, vocab = 3, 5, 11
    torch.manual_seed(6)
    base = torch.randn(batch, steps, vocab, dtype=torch.bfloat16)
    transitions = torch.randn(vocab, vocab, dtype=torch.bfloat16)
    anchors = torch.tensor([0, 1, 2])
    expected, embed_inputs, remote = [], [], []
    prev = anchors
    for step in range(steps):
        embed_inputs.append(prev.clone())
        scores = base[:, step] + transitions[prev]
        values, ids = scores[:, 5:].max(-1)
        remote.append(torch.stack([values.float(), (ids + 5).float()], -1))
        prev = scores.argmax(-1)
        expected.append(prev)
    group = SimpleNamespace(world_size=2, rank_in_group=0, unique_name="tp")
    calls = []

    def gather(pair, dim):
        assert dim == -1 and pair.dtype == torch.float32
        result = torch.cat([pair, remote[len(calls)]], -1)
        calls.append(pair)
        return result

    group.all_gather = gather
    scope["get_tp_group"] = lambda: group
    model = scope["DSparkDeepseekV41ForCausalLM"]()
    model.config = SimpleNamespace(vocab_size=vocab)
    model.lm_head = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        org_vocab_size=vocab,
        num_added_embeddings=0,
        shard_indices=SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=5),
    )
    model.draft_id_to_target_id = None
    model.compute_local_draft_logits = lambda hidden: base[:, :, :5].flatten(0, 1)
    model.markov_embed = lambda ids: ids
    model.markov_bias = lambda ids: transitions[ids]
    model.compute_confidence = lambda hidden, embed: embed.float()

    class Wrapper:
        def get_original_model(self):
            return model

        def __getattr__(self, key):
            return getattr(model, key)

    class Base:
        def _sample_sequential(self, *args):
            self.fallback = True

        def _validate_local_argmax_reduction(self):
            self.fallback = True

    spec_type = type("Spec", (scope["V41LocalArgmaxMixin"], Base), {})
    spec = spec_type()
    spec.model = Wrapper()
    spec.use_local_argmax_reduction = True
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")
    spec.num_speculative_steps = steps
    spec.sample_indices = torch.arange(batch * steps)
    spec.input_buffers = SimpleNamespace(input_ids=anchors)
    spec._anchor_idx = torch.arange(batch)
    spec.use_confidence_head = True
    spec.draft_tokens = torch.empty(batch, steps, dtype=torch.int64)
    spec.draft_token_confidence_probs = torch.empty(batch, steps)
    spec._validate_local_argmax_reduction()
    spec._sample_sequential(batch, torch.zeros(batch * steps, 2))
    torch.testing.assert_close(spec.draft_tokens, torch.stack(expected, 1))
    torch.testing.assert_close(spec.draft_token_confidence_probs, torch.stack(embed_inputs, 1).float())
    assert len(calls) == steps
    spec.use_local_argmax_reduction = False
    spec._sample_sequential(batch, torch.zeros(1))
    assert spec.fallback
    spec.use_local_argmax_reduction = True
    spec.speculative_config.draft_sample_method = "probabilistic"
    with pytest.raises(ValueError, match="greedy"):
        spec._validate_local_argmax_reduction()
    spec.speculative_config.draft_sample_method = "greedy"
    spec.model.get_original_model().lm_head.tp_size = 1
    with pytest.raises(ValueError, match="TP layout"):
        spec._validate_local_argmax_reduction()


def test_real_model_local_logits_and_graph_wrapper():
    # Requires the repository's normal CPU/NPU custom-op bootstrap.
    from vllm_ascend.models.deepseek_v41.dspark import DSparkDeepseekV41ForCausalLM
    from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext

    model = DSparkDeepseekV41ForCausalLM.__new__(DSparkDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    hidden = torch.randn(3, 4)
    norm = Mock(return_value=hidden + 1)
    logits = Mock(return_value=torch.randn(3, 7))
    model.model = SimpleNamespace(norm=norm)
    model.lm_head = object()
    model.logits_processor = logits
    out = ModelWithContext(model).compute_local_draft_logits(hidden)
    assert out is logits.return_value
    norm.assert_called_once_with(hidden)
    logits.assert_called_once_with(model.lm_head, norm.return_value, skip_gather=True)
