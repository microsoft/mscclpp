# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU tests for the native MegaMoE benchmark operand corpus."""

from types import SimpleNamespace

import pytest

from mscclpp.ext.megamoe.benchmark import _operand_corpus, _validate_corpus


def _config(rank=0, tokens=5):
    return SimpleNamespace(rank=rank, max_tokens=tokens, hidden=8, top_k=3, num_experts=8)


@pytest.mark.parametrize("tokens", [1, 2, 5])
def test_operand_corpus_is_deterministic_distinct_and_covers_token_boundaries(tokens):
    torch = pytest.importorskip("torch")
    first = _operand_corpus(_config(tokens=tokens), torch.device("cpu"), 1234, 20)
    second = _operand_corpus(_config(tokens=tokens), torch.device("cpu"), 1234, 20)
    assert len(first) == 20
    for left, right in zip(first, second):
        for actual, expected in zip(left[:3], right[:3]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert tuple(left[0].shape) == (tokens, 8)
        assert tuple(left[1].shape) == tuple(left[2].shape) == (tokens, 3)
        assert left[3].shape == left[0].shape
        assert len({tensor.data_ptr() for tensor in left}) == 4
    for field in range(4):
        assert len({operand[field].data_ptr() for operand in first}) == 20
    # Routing cases alternate and revisit; the complete operands remain unique.
    assert torch.equal(first[0][1], first[2][1])
    assert not torch.equal(first[0][0], first[2][0])
    assert not torch.equal(first[0][2], first[2][2])
    assert len({tuple(operand[0].flatten().tolist()) for operand in first}) == 20


def test_operand_corpus_is_rank_asymmetric():
    torch = pytest.importorskip("torch")
    rank0 = _operand_corpus(_config(rank=0), torch.device("cpu"), 17, 4)
    rank1 = _operand_corpus(_config(rank=1), torch.device("cpu"), 17, 4)
    for zero, one in zip(rank0, rank1):
        assert not torch.equal(zero[0], one[0])
        assert not torch.equal(zero[1], one[1])
        assert not torch.equal(zero[2], one[2])


def test_validate_corpus_checks_every_association_and_rejects_stale_cache():
    torch = pytest.importorskip("torch")
    outputs = [torch.full((2, 4), value, dtype=torch.bfloat16) for value in (1.0, 2.0, 1.0, 3.0)]
    corpus = [(value, torch.empty(0), torch.empty(0), value.clone()) for value in outputs]
    report = _validate_corpus(corpus, outputs, rtol=0, atol=0)
    assert report == {
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "outputs_checked": 4,
        "stale_output_rejected": True,
    }
    corpus[2] = (*corpus[2][:3], outputs[1].clone())
    with pytest.raises(AssertionError):
        _validate_corpus(corpus, outputs, rtol=0, atol=0)


def test_validate_corpus_rejects_indistinguishable_neighbor_outputs():
    torch = pytest.importorskip("torch")
    output = torch.ones((1, 4), dtype=torch.bfloat16)
    corpus = [(output, output, output, output.clone()) for _ in range(2)]
    with pytest.raises(AssertionError, match="stale cached output"):
        _validate_corpus(corpus, [output, output.clone()], rtol=0, atol=0)


def test_operand_corpus_rejects_empty_count():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="must not be empty"):
        _operand_corpus(_config(), torch.device("cpu"), 1, 0)
