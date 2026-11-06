"""Educational single-head self-attention, built manually from Q, K, V.

Paper: Vaswani et al., 2017, "Attention Is All You Need", Eq. 1:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

For self-attention, Q, K, V are linear projections of the same input
token sequence X (paper's W^Q, W^K, W^V, no bias term):

    Q = X W^Q,   K = X W^K,   V = X W^V

Symbol -> code variable -> shape -> meaning:
    X (input tokens)       x                   [B,N,D]   token embeddings (B=batch, N=#tokens, D=embed dim)
    W^Q, W^K               W_q.weight, W_k.weight  [d_k,D]   learned query/key projections (nn.Linear convention: out_features x in_features)
    W^V                    W_v.weight          [d_v,D]   learned value projection
    Q, K                   q, k                [B,N,d_k] projected queries/keys
    V                      v                   [B,N,d_v] projected values
    Q K^T                  raw_scores          [B,N,N]   raw_scores[b,i,j] = q[b,i] . k[b,j], similarity between query i and key j
    sqrt(d_k)              scale               scalar    stabilizes softmax gradients (paper Sec. 3.2.1); without it, large d_k makes dot products large and softmax saturates
    softmax(.)             attention_weights   [B,N,N]   row i sums to 1: how much query token i attends to each key token j
    softmax(.) V           output              [B,N,d_v] output[b,i] = weighted sum over j of attention_weights[b,i,j] * v[b,j]

Indexing convention: N is the sequence length (number of tokens), shared
by queries and keys/values (plain self-attention, no cross-attention,
no causal mask -- every token can attend to every other token,
including itself). Batch dimension B is carried through unchanged;
all matrix products below use torch.bmm/matmul semantics where the
leading [B] dimension is batched independently.

This file is deliberately "manual": Q/K/V come from nn.Linear
projections (that part is standard and would be reinvented pointlessly
otherwise), but the scaled dot product, the softmax, and the weighted
sum over V are all written out explicitly (no nn.MultiheadAttention,
no F.scaled_dot_product_attention, no F.softmax) so every intermediate
tensor and shape is visible.

Local usage (lightweight, CPU, no dataset, no GPU):
    python code/transformers/self_attention_qkv.py --smoke-test
"""

import argparse
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def manual_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax(scores)_j = exp(scores_j) / sum_j' exp(scores_j'), written
    out explicitly (no F.softmax) with the standard max-subtraction
    trick for numerical stability: subtracting a constant (the row max)
    from every logit before exponentiating does not change the result
    (it cancels in the ratio) but prevents exp() from overflowing.

    Args:
        scores: [...,N] raw scores, softmax taken over the last axis by default
        dim:    axis to normalize over

    Returns:
        weights: same shape as scores, non-negative, sums to 1 along `dim`
    """
    scores_shifted = scores - scores.amax(dim=dim, keepdim=True)  # same shape; subtract per-row max for stability
    exp_scores = torch.exp(scores_shifted)
    weights = exp_scores / exp_scores.sum(dim=dim, keepdim=True)  # normalize so each row sums to 1
    return weights


class SelfAttentionQKV(nn.Module):
    """Single-head scaled dot-product self-attention, Eq. 1 of
    Vaswani et al., 2017, applied as self-attention (Q, K, V all
    derived from the same input X)."""

    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        # Paper's W^Q, W^K, W^V have no bias term -- bias=False matches Eq. 1 exactly.
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)  # weight: [head_dim, embed_dim]
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)  # weight: [head_dim, embed_dim]
        self.W_v = nn.Linear(embed_dim, head_dim, bias=False)  # weight: [head_dim, embed_dim]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B,N,D] input token embeddings, D == self.embed_dim

        Returns a dict of every named intermediate tensor (for printing
        shapes and for the worked example / visualization below):
            q, k, v:            [B,N,d_k] (d_k == d_v == self.head_dim here)
            raw_scores:         [B,N,N], Q K^T before scaling
            scaled_scores:      [B,N,N], raw_scores / sqrt(d_k)
            attention_weights:  [B,N,N], softmax(scaled_scores), rows sum to 1
            output:             [B,N,d_v]
        """
        batch_size, num_tokens, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        q = self.W_q(x)  # [B,N,D] -> [B,N,d_k], linear projection applied independently per token
        k = self.W_k(x)  # [B,N,D] -> [B,N,d_k]
        v = self.W_v(x)  # [B,N,D] -> [B,N,d_v]

        # Q K^T: for each batch b, [N,d_k] @ [d_k,N] -> [N,N]. k.transpose(-2,-1)
        # swaps the last two axes [B,N,d_k] -> [B,d_k,N] so matmul contracts
        # over d_k, the shared feature axis, leaving [B,N,N]: entry (i,j) is
        # the dot product of query i with key j.
        raw_scores = torch.matmul(q, k.transpose(-2, -1))  # [B,N,d_k] @ [B,d_k,N] -> [B,N,N]

        scale = math.sqrt(self.head_dim)  # sqrt(d_k), scalar
        scaled_scores = raw_scores / scale  # [B,N,N], unchanged shape

        attention_weights = manual_softmax(scaled_scores, dim=-1)  # [B,N,N], softmax over keys (last axis) per query row

        # attention_weights @ v: [B,N,N] @ [B,N,d_v] -> [B,N,d_v]; contracts
        # over the key axis N, so output[b,i] = sum_j attention_weights[b,i,j] * v[b,j].
        output = torch.matmul(attention_weights, v)  # [B,N,N] @ [B,N,d_v] -> [B,N,d_v]

        return {
            "q": q,
            "k": k,
            "v": v,
            "raw_scores": raw_scores,
            "scaled_scores": scaled_scores,
            "attention_weights": attention_weights,
            "output": output,
        }


def print_all_shapes(x: torch.Tensor, intermediates: dict[str, torch.Tensor]) -> None:
    """Prints the shape of the input and every intermediate tensor, in
    the order they are produced by SelfAttentionQKV.forward."""
    print(f"x (input tokens)         : {tuple(x.shape)}")
    print(f"q (= X W^Q)              : {tuple(intermediates['q'].shape)}")
    print(f"k (= X W^K)              : {tuple(intermediates['k'].shape)}")
    print(f"v (= X W^V)              : {tuple(intermediates['v'].shape)}")
    print(f"raw_scores (= Q K^T)     : {tuple(intermediates['raw_scores'].shape)}")
    print(f"scaled_scores (/sqrt(d_k)): {tuple(intermediates['scaled_scores'].shape)}")
    print(f"attention_weights        : {tuple(intermediates['attention_weights'].shape)}")
    print(f"output (= weights @ V)   : {tuple(intermediates['output'].shape)}")


def make_toy_token_sequence(seed: int = 0) -> tuple[torch.Tensor, list[str]]:
    """Hand-crafted toy "sentence" with category structure, so the
    (untrained, randomly-initialized) attention heatmap below is more
    interpretable than pure random noise would be: tokens from the same
    semantic category share a base embedding direction plus small
    independent noise, so their raw dot products are more similar to
    each other than to tokens from a different category.

    Returns:
        x:      [1,N,D] single-sequence batch, D=8
        tokens: list of N token labels, for axis ticks / printing
    """
    generator = torch.Generator().manual_seed(seed)
    embed_dim = 8

    # NOTE on interpretation: W^Q/W^K/W^V below are randomly initialized
    # and never trained, so the resulting attention pattern is NOT
    # expected to group tokens by the semantic categories defined here --
    # a random projection does not reliably preserve which input
    # directions are "similar" in a category sense. The categories exist
    # only to give x structure beyond pure noise; the attention heatmap
    # demonstrates the Eq. 1 MECHANISM (shapes, softmax normalization,
    # weighted combination of V), not learned semantic attention.
    tokens = ["cat", "dog", "car", "truck", "apple", "banana"]
    category_base = {
        "cat": 0,
        "dog": 0,  # animals: category 0
        "car": 1,
        "truck": 1,  # vehicles: category 1
        "apple": 2,
        "banana": 2,  # fruits: category 2
    }
    num_categories = 3
    base_vectors = torch.randn(num_categories, embed_dim, generator=generator) * 2.0  # [3,D], well-separated category directions

    noise_scale = 0.3
    rows = []
    for token in tokens:
        category_id = category_base[token]
        noise = torch.randn(embed_dim, generator=generator) * noise_scale  # [D]
        rows.append(base_vectors[category_id] + noise)  # [D]
    x = torch.stack(rows, dim=0).unsqueeze(0)  # [N,D] -> [1,N,D], batch size 1
    return x, tokens


def visualize_attention_matrix(
    attention_weights: torch.Tensor,
    tokens: list[str],
    save_path: str | None = None,
):
    """Heatmap of one example's attention matrix.

    Args:
        attention_weights: [N,N] (already indexed to a single batch element)
        tokens:             length-N list of token labels for both axes
        save_path:          if given, save instead of showing interactively
    """
    num_tokens = len(tokens)
    assert attention_weights.shape == (num_tokens, num_tokens)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(attention_weights.detach().numpy(), cmap="viridis", vmin=0.0, vmax=attention_weights.max().item())
    ax.set_xticks(range(num_tokens))
    ax.set_yticks(range(num_tokens))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("key token j")
    ax.set_ylabel("query token i")
    ax.set_title("attention_weights[i, j] = softmax(Q K^T / sqrt(d_k))", fontsize=9)
    fig.colorbar(im, ax=ax, label="attention weight", fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved attention-matrix figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def worked_example_reconstruct_output(
    v: torch.Tensor,
    attention_weights: torch.Tensor,
    output: torch.Tensor,
    tokens: list[str],
    query_index: int,
) -> torch.Tensor:
    """Concrete worked example using the attention matrix: for one query
    token, print its attention weight onto every key token (sorted,
    most-attended first), then manually reconstruct that query's output
    vector as the explicit weighted sum of value vectors
        output[query_index] = sum_j attention_weights[query_index, j] * v[j]
    and check it against the module's own output row.

    Args:
        v:                 [N,d_v]
        attention_weights: [N,N]
        output:            [N,d_v]
        tokens:             length-N token labels
        query_index:       which query row (0-indexed) to explain

    Returns:
        manual_output_row: [d_v], the hand-computed reconstruction (for the caller to assert on)
    """
    num_tokens, head_dim = v.shape
    weights_row = attention_weights[query_index]  # [N], attention from this query to every key

    print(f"\nWorked example: query token = '{tokens[query_index]}' (index {query_index})")
    print("attention weights onto each key token, sorted by weight (descending):")
    sorted_indices = torch.argsort(weights_row, descending=True)
    for key_index in sorted_indices.tolist():
        print(f"  -> '{tokens[key_index]:<8s}' (index {key_index}): weight = {weights_row[key_index].item():.4f}")

    # Manual reconstruction: explicit weighted sum over key/value tokens,
    # rather than the batched torch.matmul used inside forward(). weights_row
    # is [N]; v is [N,d_v]; weights_row.unsqueeze(-1) -> [N,1] broadcasts
    # against v's [N,d_v] so each value row v[j] is scaled by its own
    # scalar weight before summing over j (dim=0) -> [d_v].
    manual_output_row = (weights_row.unsqueeze(-1) * v).sum(dim=0)  # [N,1]*[N,d_v] -> [N,d_v], summed over N -> [d_v]

    module_output_row = output[query_index]  # [d_v]
    max_abs_difference = (manual_output_row - module_output_row).abs().max().item()
    print(
        f"manual reconstruction vs. module output, max |difference| = {max_abs_difference:.3e} "
        f"({'matches' if max_abs_difference < 1e-5 else 'MISMATCH'})"
    )
    return manual_output_row


def _smoke_test():
    """Lightweight local checks: shapes at every step, manual_softmax
    exactly matches torch.softmax, attention rows sum to 1, finite
    values, gradients flow back to W_q/W_k/W_v, fixed-seed
    reproducibility, and the worked-example reconstruction exactly
    matches the module's own output row. No dataset, no GPU."""
    torch.manual_seed(0)
    batch_size, num_tokens, embed_dim, head_dim = 2, 5, 8, 4  # embed_dim=8 matches make_toy_token_sequence below

    x = torch.randn(batch_size, num_tokens, embed_dim, requires_grad=True)
    module = SelfAttentionQKV(embed_dim, head_dim)

    intermediates = module(x)
    q, k, v = intermediates["q"], intermediates["k"], intermediates["v"]
    scaled_scores = intermediates["scaled_scores"]
    attention_weights = intermediates["attention_weights"]
    output = intermediates["output"]

    assert q.shape == (batch_size, num_tokens, head_dim)
    assert k.shape == (batch_size, num_tokens, head_dim)
    assert v.shape == (batch_size, num_tokens, head_dim)
    assert intermediates["raw_scores"].shape == (batch_size, num_tokens, num_tokens)
    assert scaled_scores.shape == (batch_size, num_tokens, num_tokens)
    assert attention_weights.shape == (batch_size, num_tokens, num_tokens)
    assert output.shape == (batch_size, num_tokens, head_dim)

    # manual_softmax must exactly match the reference implementation.
    reference_weights = torch.softmax(scaled_scores, dim=-1)
    assert torch.allclose(attention_weights, reference_weights, atol=1e-6)

    # Every attention row is a probability distribution over keys.
    row_sums = attention_weights.sum(dim=-1)  # [B,N]
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    assert torch.isfinite(output).all()

    # Gradients must flow back to all three projection matrices.
    loss = output.sum()
    loss.backward()
    for name, param in [("W_q", module.W_q.weight), ("W_k", module.W_k.weight), ("W_v", module.W_v.weight)]:
        assert param.grad is not None, f"{name}.weight has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name}.weight gradient is not finite"

    # Fixed-seed reproducibility: same module (no randomness in forward), same input -> identical output.
    with torch.no_grad():
        output_repeat = module(x)["output"]
    assert torch.equal(output, output_repeat)

    # Worked example: manual reconstruction must exactly match the module's computed output row.
    x_toy, tokens = make_toy_token_sequence(seed=0)
    with torch.no_grad():
        toy_intermediates = module(x_toy)
    manual_row = worked_example_reconstruct_output(
        toy_intermediates["v"][0],
        toy_intermediates["attention_weights"][0],
        toy_intermediates["output"][0],
        tokens,
        query_index=0,
    )
    assert torch.allclose(manual_row, toy_intermediates["output"][0, 0], atol=1e-5)

    print("\nself_attention_qkv smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run lightweight local checks and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-dim", type=int, default=4)
    parser.add_argument("--query-index", type=int, default=0, help="which toy token to use as the worked-example query")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        torch.manual_seed(args.seed)
        x, tokens = make_toy_token_sequence(seed=args.seed)  # x: [1,N,8]
        embed_dim = x.shape[-1]

        module = SelfAttentionQKV(embed_dim=embed_dim, head_dim=args.head_dim)
        intermediates = module(x)

        print_all_shapes(x, intermediates)

        attention_weights = intermediates["attention_weights"][0]  # [N,N], single batch element
        worked_example_reconstruct_output(
            intermediates["v"][0],
            attention_weights,
            intermediates["output"][0],
            tokens,
            query_index=args.query_index,
        )
        visualize_attention_matrix(attention_weights, tokens)
